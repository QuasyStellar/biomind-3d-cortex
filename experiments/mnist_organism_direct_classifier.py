"""
Архитектурное изменение (по прямой просьбе "попробуй архитектурные
изменения", после того как перебор гиперпараметров вокруг linear-probe
подхода (`mnist_organism_classification.py`, 12%->63.3%) начал давать
убывающую отдачу): вместо self-supervised представления + Hebbian
associative read-out - ОТДЕЛЬНЫЙ "классификатор-геном" (тот же класс
`PredictiveCodingNet`, zero-backward PC-релаксация, что и everywhere в
проекте), обучаемый НАПРЯМУЮ на классификацию, с локальным контекстом
каждой клетки (48-мерным, тем же, что self-supervised геном) как входом,
и УСРЕДНЕНИЕМ голосов по клеткам как выходом - идея близка к M8-колончатому
голосованию (docs/VERIFICATION_LOG.md), но здесь каждая "колонка" - это
живая клетка со своим локальным видом на цифру, не отдельная модальность.

ВАЖНАЯ НАХОДКА ПО ХОДУ: первая версия (train_step на СЫРЫХ per-cell
контекстах, batch = все ~113 клеток ОДНОЙ цифры) дала тот же вырожденный
коллапс (8%, всегда один класс), что и в самом начале linear-probe
диагностики - причина другая: 113 строк батча внутри ОДНОЙ цифры почти
идентичны (локальный контекст соседних клеток сильно коррелирован),
батч НЕ разнообразен по классам, а PC-релаксация (как и в
`m_minus1_symmetric_comparison.py`/`mnist_pc_vs_backprop_sanity.py`)
рассчитана на батчи со СМЕШАННЫМИ классами. ИСПРАВЛЕНО: один POOLED
(2x2 пространственная сетка, как в linear-probe версии) признак НА
ЦИФРУ, батчи собираются из МНОГИХ РАЗНЫХ цифр (как в уже проверенном
протоколе M(-1)/MNIST) - сразу дало неколлапсирующий результат.
"""
import sys, os, time, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from core.unified_organism import LivingTissue
from core.predictive_coding import PredictiveCodingNet
from core.mnist_loader import load_mnist
import experiments.mnist_organism_classification as base

SIZE = base.SIZE
DIGIT_SIDE = base.DIGIT_SIDE
GROWTH_STEPS = base.GROWTH_STEPS
BOOTSTRAP_CYCLES = base.BOOTSTRAP_CYCLES
BOOTSTRAP_STEPS = base.BOOTSTRAP_STEPS
BOOTSTRAP_FRACTION = base.BOOTSTRAP_FRACTION
GENOME_HIDDEN = 128
K_STEPS_PER_DIGIT = 30
POOL = 2
N_TRAIN = 600
N_TEST = 300
CLASSIFIER_HIDDEN = 128
EPOCHS = 150
BATCH = 32


def pooled_ctx(snapshot, image, k_steps=K_STEPS_PER_DIGIT, pool=POOL):
    org = copy.deepcopy(snapshot)
    org.growth_enabled = True
    for _ in range(k_steps):
        org.step(sensory_signal=base.digit_signal(image), train_genome=False)
    ctx_flat, ys, xs = org.compute_context()
    cdim = ctx_flat.shape[1]
    off = (SIZE - DIGIT_SIDE) // 2
    grid = torch.zeros(cdim, SIZE, SIZE)
    grid[:, ys, xs] = ctx_flat.T
    patch = grid[:, off:off + DIGIT_SIDE, off:off + DIGIT_SIDE]
    return F.adaptive_avg_pool2d(patch.unsqueeze(0), pool).flatten()


def run(seed=1):
    tr_x, tr_y, te_x, te_y = load_mnist()
    torch.manual_seed(seed)
    organism = LivingTissue(size=SIZE, state_dim=16, seed=seed, genome_hidden=GENOME_HIDDEN)

    t0 = time.time()
    for t in range(GROWTH_STEPS):
        n, err = organism.step(sensory_signal=base.blob_signal(t), train_genome=True)
    for cycle in range(BOOTSTRAP_CYCLES):
        organism.ablate(fraction=BOOTSTRAP_FRACTION)
        for t in range(BOOTSTRAP_STEPS):
            n, err = organism.step(sensory_signal=base.blob_signal(t), train_genome=True)
    print(f"Популяция раскручена: {n} живых клеток ({time.time()-t0:.1f}s)")
    organism.growth_enabled = False
    snapshot = copy.deepcopy(organism)

    context_dim = organism.state_dim * 3
    feat_dim = context_dim * POOL * POOL
    classifier = PredictiveCodingNet([feat_dim, CLASSIFIER_HIDDEN, 10], relax_steps=15,
                                      relax_lr=0.08, weight_lr=0.01, seed=seed, adam=True,
                                      weight_decay=0.02)

    g = torch.Generator().manual_seed(seed + 1)
    train_idx = torch.randperm(tr_x.shape[0], generator=g)[:N_TRAIN].tolist()
    test_idx = torch.randperm(te_x.shape[0], generator=torch.Generator().manual_seed(seed + 2))[:N_TEST].tolist()

    print(f"Вычисление признаков: {N_TRAIN} train + {N_TEST} test цифр (K={K_STEPS_PER_DIGIT}, pool={POOL})...")
    t0 = time.time()
    train_feats, train_labels = [], []
    for count, i in enumerate(train_idx):
        train_feats.append(pooled_ctx(snapshot, tr_x[i]))
        train_labels.append(tr_y[i].item())
        if (count + 1) % 200 == 0:
            print(f"  train {count+1}/{N_TRAIN} ({time.time()-t0:.1f}s)")
    train_feats = torch.stack(train_feats)
    train_labels = torch.tensor(train_labels)

    test_feats, test_labels = [], []
    for count, i in enumerate(test_idx):
        test_feats.append(pooled_ctx(snapshot, te_x[i]))
        test_labels.append(te_y[i].item())
        if (count + 1) % 200 == 0:
            print(f"  test {count+1}/{N_TEST} ({time.time()-t0:.1f}s)")
    test_feats = torch.stack(test_feats)
    test_labels = torch.tensor(test_labels)
    print(f"Признаки готовы ({time.time()-t0:.1f}s)")

    print(f"Обучение классификатора: {EPOCHS} эпох, batch={BATCH}...")
    t0 = time.time()
    gb = torch.Generator().manual_seed(seed + 5)
    for epoch in range(EPOCHS):
        perm = torch.randperm(N_TRAIN, generator=gb)
        for b0 in range(0, N_TRAIN, BATCH):
            idx_b = perm[b0:b0 + BATCH]
            xb = train_feats[idx_b]
            yb = torch.zeros(len(idx_b), 10)
            yb.scatter_(1, train_labels[idx_b].unsqueeze(1), 1.0)
            classifier.train_step(xb, yb)
        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                pred = classifier.forward_pass(train_feats).argmax(dim=1)
            train_acc = (pred == train_labels).float().mean().item()
            with torch.no_grad():
                pred_te = classifier.forward_pass(test_feats).argmax(dim=1)
            test_acc = (pred_te == test_labels).float().mean().item()
            print(f"  epoch {epoch+1}/{EPOCHS}  train_acc={train_acc*100:.1f}%  "
                  f"test_acc={test_acc*100:.1f}%  ({time.time()-t0:.1f}s)")

    with torch.no_grad():
        pred = classifier.forward_pass(test_feats).argmax(dim=1)
    acc = (pred == test_labels).float().mean().item()
    print("=" * 70)
    print(f"ИТОГ: accuracy={acc*100:.1f}%  (N_test={N_TEST}, chance=10%)")
    print("По классам:")
    for c in range(10):
        mask = test_labels == c
        tot = int(mask.sum().item())
        cor = int((pred[mask] == c).sum().item())
        pct = 100.0 * cor / tot if tot > 0 else float("nan")
        print(f"  {c}: {cor}/{tot} ({pct:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run()

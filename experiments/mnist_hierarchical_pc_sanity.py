"""
Второй уровень PC-релаксации над pooled-сеткой (core/hierarchical_pc.py) -
следующая архитектурная идея из VERIFICATION_LOG после того, как ШЕСТЬ
попыток исправить ПРОВОДКУ между клетками (feedback alignment, temporal PC,
small-world, таламический роутер, fire_rate, 2D vs 3D) и ДВЕ попытки
top-down label bias в САМ ГЕНОМ не сдвинули MNIST-through-tissue дальше
69.0%. Здесь - не проводка и не геном, а способ АГРЕГАЦИИ per-cell
предсказаний в финальный признак: вместо плоского SPATIAL_POOL (усреднение
без учёта отношений между регионами) - self-supervised, локально-связная
(3x3 по сетке регионов) вторая PC-сеть поверх baseline-вычтенного L1-
признака (та же анти-коллапс дисциплина, что и в основном пайплайне).

Фазы (та же дисциплина заморозки/снапшотов, что и в mnist_organism_classification.py
и mnist_topdown_bias_sanity.py - НЕ повторяем уже найденный баг дрейфа
весов при последовательном обучении на многих примерах подряд):
1. Bootstrap L1-ткани (как в base), заморозка (train_genome=False).
2. L2-претрейн (self-supervised, ОТДЕЛЬНАЯ фаза): на N_PRETRAIN train-цифрах
   обучаем HierarchicalPC реконструировать baseline-вычтенный L1-признак,
   затем СНАПШОТ (deepcopy) - веса L2 замораживаются перед classification.
3. Linear-probe (как в base), но признак = L2.encode(diff), не сырой
   pooled diff.
"""
import sys, os, time, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from core.unified_organism import LivingTissue
from core.mnist_loader import load_mnist
from core.hierarchical_pc import HierarchicalPC
import experiments.mnist_organism_classification as base

H2_DIM = 32          # каналов на позицию сетки в bottleneck L2 (h_dim=GENOME_HIDDEN=128 -> 32, 4x сжатие)
N_PRETRAIN = 400      # цифр для self-supervised претрейна L2
PRETRAIN_BATCH = 16


def raw_hidden_grid(org, image, k_steps, pool=base.SPATIAL_POOL):
    """Как base.raw_hidden, но возвращает НЕ сплющенный (gh, pool, pool)
    тензор - нужен gh явно для локальной маски L2, а не только итоговый
    плоский вектор."""
    org.growth_enabled = True
    for _ in range(k_steps):
        org.step(sensory_signal=base.digit_signal(image), train_genome=False)
    ctx_flat, ys, xs = org.compute_context()
    h = org.hidden_representation(ctx_flat)
    off = (base.SIZE - base.DIGIT_SIDE) // 2
    gh = h.shape[1]
    grid = torch.zeros(gh, base.SIZE, base.SIZE)
    grid[:, ys, xs] = h.T
    mask = torch.zeros(1, base.SIZE, base.SIZE)
    mask[:, ys, xs] = 1.0
    val_patch = grid[:, off:off + base.DIGIT_SIDE, off:off + base.DIGIT_SIDE]
    mask_patch = mask[:, off:off + base.DIGIT_SIDE, off:off + base.DIGIT_SIDE]
    val_pooled = F.adaptive_avg_pool2d(val_patch.unsqueeze(0), pool)
    mask_pooled = F.adaptive_avg_pool2d(mask_patch.unsqueeze(0), pool)
    return (val_pooled / (mask_pooled + 1e-6)).squeeze(0)  # (gh, pool, pool)


def run(seed=1, n_train=400, n_test=200, n_pretrain=N_PRETRAIN, h2_dim=H2_DIM):
    tr_x, tr_y, te_x, te_y = load_mnist()
    torch.manual_seed(seed)
    organism = LivingTissue(size=base.SIZE, state_dim=16, seed=seed,
                             genome_hidden=base.GENOME_HIDDEN, fire_rate=base.FIRE_RATE)

    t0 = time.time()
    for t in range(base.GROWTH_STEPS):
        n, err = organism.step(sensory_signal=base.blob_signal(t), train_genome=True)
    for cycle in range(base.BOOTSTRAP_CYCLES):
        organism.ablate(fraction=base.BOOTSTRAP_FRACTION)
        for t in range(base.BOOTSTRAP_STEPS):
            n, err = organism.step(sensory_signal=base.blob_signal(t), train_genome=True)
    organism.growth_enabled = False
    print(f"Bootstrap завершён: {n} клеток ({time.time()-t0:.1f}s)")

    K = base.K_STEPS_PER_DIGIT
    pool = base.SPATIAL_POOL
    baseline_grid = raw_hidden_grid(copy.deepcopy(organism), torch.zeros(base.DIGIT_SIDE, base.DIGIT_SIDE), K, pool)
    baseline_flat = baseline_grid.flatten()
    gh = baseline_grid.shape[0]

    hpc = HierarchicalPC(pool=pool, h_dim=gh, h2_dim=h2_dim, radius=1, seed=seed + 500)

    g0 = torch.Generator().manual_seed(seed + 20)
    pretrain_idx = torch.randperm(tr_x.shape[0], generator=g0)[:n_pretrain].tolist()
    t0 = time.time()
    buf = []
    for count, i in enumerate(pretrain_idx):
        image = tr_x[i]
        diff = raw_hidden_grid(copy.deepcopy(organism), image, K, pool).flatten() - baseline_flat
        buf.append(diff)
        if len(buf) == PRETRAIN_BATCH:
            energy = hpc.train_step(torch.stack(buf))
            buf = []
        if (count + 1) % 100 == 0:
            print(f"  L2 претрейн {count+1}/{n_pretrain} ({time.time()-t0:.1f}s)")
    if buf:
        hpc.train_step(torch.stack(buf))
    print(f"L2 претрейн завершён ({time.time()-t0:.1f}s)")
    hpc_snapshot = copy.deepcopy(hpc)  # заморозка L2 - та же дисциплина, что и для L1-ткани

    def feature_for_digit(image):
        org = copy.deepcopy(organism)
        diff = raw_hidden_grid(org, image, K, pool).flatten() - baseline_flat
        feat = hpc_snapshot.encode(diff.unsqueeze(0)).squeeze(0)
        return F.normalize(feat, dim=0)

    organism.add_modality("mnist_readout", key_dim=h2_dim * pool * pool, value_dim=10, seed=99,
                           sdr_dim=base.SDR_DIM, sparsity=base.SDR_SPARSITY)

    g1 = torch.Generator().manual_seed(seed + 1)
    train_idx = torch.randperm(tr_x.shape[0], generator=g1)[:n_train].tolist()
    test_idx = torch.randperm(te_x.shape[0], generator=torch.Generator().manual_seed(seed + 2))[:n_test].tolist()

    t0 = time.time()
    for count, i in enumerate(train_idx):
        image, label = tr_x[i], tr_y[i].item()
        feat = feature_for_digit(image)
        onehot = torch.zeros(10)
        onehot[label] = 1.0
        organism.write_fact_modal("mnist_readout", feat, onehot, tag_strength=1.0)
        if (count + 1) % 100 == 0:
            print(f"  linear-probe обучение {count+1}/{n_train} ({time.time()-t0:.1f}s)")
    print(f"Linear-probe обучение завершено: {n_train} цифр ({time.time()-t0:.1f}s)")

    correct = 0
    for i in test_idx:
        image, label = te_x[i], te_y[i].item()
        feat = feature_for_digit(image)
        pred = organism.read_fact_modal("mnist_readout", feat)
        correct += int(pred.argmax().item() == label)
    acc = correct / n_test
    print(f"ИТОГ: accuracy={acc*100:.1f}%  (N_test={n_test}, chance=10%)")
    return acc


if __name__ == "__main__":
    run()

"""
Атака на корень проблемы (не проводку между клетками, а сам геном) -
слабость №2 из находок Гермеса: "геном учится слепому 3x3 сглаживанию,
не зная семантики задачи". Шесть попыток исправить ПРОВОДКУ (feedback
alignment, temporal PC, small-world, таламический роутер, fire_rate,
2D vs 3D) не сдвинули MNIST-through-tissue дальше 69.0% - эта попытка
целится в саму КЛЕТКУ, не в связи между ними.

Идея (Rao & Ballard 1999, Friston - top-down predictions уже цитировались
в спецификации Гермеса, но никогда не реализовывались): вместо того,
чтобы прикручивать классификатор СНАРУЖИ поверх self-supervised
представления (linear-probe подход, уже давший 69.0%), даём ГЕНОМУ
самому предсказывать метку класса КАК ЧАСТЬ своей обычной задачи
самопредсказания - one-hot метка транслируется (broadcast) во ВСЕ живые
клетки как ДОПОЛНИТЕЛЬНЫЕ "химические" каналы (top-down bias) во время
СПЕЦИАЛЬНОЙ фазы обучения на реальных цифрах (после self-supervised
bootstrap, до заморозки). Геном учится предсказывать identity клетки
ПО КОНТЕКСТУ СОСЕДЕЙ, включающему теперь и label-каналы соседей - если
идея верна, локальные визуальные паттерны конкретных цифр должны
"впечататься" в веса генома вместе с их меткой, не только с абстрактной
формой.

ВАЖНО - не повторяет старый баг (дрейф весов): здесь ЕСТЬ отдельная фаза
"supervised-обучения" (train_genome=True, много цифр подряд, веса
СОЗНАТЕЛЬНО меняются под задачу), но она полностью ОТДЕЛЕНА от финальной
linear-probe фазы (веса ЗАМОРАЖИВАЮТСЯ после), и признаки для
классификации извлекаются с ЕДИНОГО замороженного снапшота для ВСЕХ
train/test цифр - та же дисциплина, что уже исправила дрейф весов раньше.
На test-время label, разумеется, НЕ подаётся (иначе жульничество) -
проверяем, стали ли ПРИЗНАКИ ткани более классифицирующими из-за того,
что геном video top-down сигнал ВО ВРЕМЯ обучения, не из-за прямой подачи
метки на инференсе.
"""
import sys, os, time, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from core.unified_organism import LivingTissue
from core.mnist_loader import load_mnist
import experiments.mnist_organism_classification as base

N_LABEL_CH = 10
SUPERVISED_DIGITS = 400  # сколько цифр показать в фазе top-down обучения
SUPERVISED_K = 20        # шагов релаксации на цифру в этой фазе (веса УЧАТСЯ)


def inject_label_broadcast(organism, label, localized=False, intensity=1.0):
    """Записывает one-hot метку НАПРЯМУЮ в state (не через sensory_signal,
    т.к. у state только 2 "сенсорных" канала 2:4 - метка получает СВОИ
    каналы 4:14, отдельные от визуального сигнала).

    localized=False (исходная версия - широковещательно по ВСЕМУ холсту,
    дала 37.7%, ХУЖЕ baseline) vs localized=True (найдено по ходу - только
    внутри патча цифры, не по всему холсту - гипотеза: broadcast-версия
    была "слишком лёгкой" задачей, доминировавшей над визуальным сигналом,
    локальная версия заставляет геном связывать метку ИМЕННО с локальным
    визуальным контекстом, не просто копировать константу). intensity<1.0
    - ослабленный сигнал, чтобы не доминировать над визуальным."""
    alive, _ = organism.alive_mask()
    onehot = torch.zeros(N_LABEL_CH)
    onehot[label] = intensity
    if localized:
        mask = torch.zeros(1, organism.size, organism.size)
        off = (organism.size - base.DIGIT_SIDE) // 2
        mask[0, off:off + base.DIGIT_SIDE, off:off + base.DIGIT_SIDE] = 1.0
        gate = alive[0, 0].float() * mask[0]
    else:
        gate = alive[0, 0].float()
    organism.state[0, 4:4 + N_LABEL_CH] = onehot.view(N_LABEL_CH, 1, 1) * gate


def run(seed=1, n_train=600, n_test=300, localized=False, intensity=1.0):
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
    print(f"Bootstrap завершён: {n} клеток ({time.time()-t0:.1f}s)")

    # ФАЗА TOP-DOWN ОБУЧЕНИЯ: веса генома АКТИВНО учатся на (визуал, метка)
    g = torch.Generator().manual_seed(seed + 10)
    sup_idx = torch.randperm(tr_x.shape[0], generator=g)[:SUPERVISED_DIGITS].tolist()
    t0 = time.time()
    for count, i in enumerate(sup_idx):
        image, label = tr_x[i], tr_y[i].item()
        for _ in range(SUPERVISED_K):
            inject_label_broadcast(organism, label, localized=localized, intensity=intensity)
            organism.step(sensory_signal=base.digit_signal(image), train_genome=True)
        if (count + 1) % 100 == 0:
            print(f"  top-down обучение {count+1}/{SUPERVISED_DIGITS} ({time.time()-t0:.1f}s)")
    organism.growth_enabled = False
    snapshot = copy.deepcopy(organism)
    print(f"Top-down фаза завершена ({time.time()-t0:.1f}s)")

    K = base.K_STEPS_PER_DIGIT
    baseline = base.raw_hidden(copy.deepcopy(snapshot), torch.zeros(base.DIGIT_SIDE, base.DIGIT_SIDE), K)
    organism.add_modality("mnist_readout", key_dim=baseline.numel(), value_dim=10, seed=99,
                           sdr_dim=base.SDR_DIM, sparsity=base.SDR_SPARSITY)

    g2 = torch.Generator().manual_seed(seed + 1)
    train_idx = torch.randperm(tr_x.shape[0], generator=g2)[:n_train].tolist()
    test_idx = torch.randperm(te_x.shape[0], generator=torch.Generator().manual_seed(seed + 2))[:n_test].tolist()

    t0 = time.time()
    for i in train_idx:
        image, label = tr_x[i], tr_y[i].item()
        h = base.raw_hidden(copy.deepcopy(snapshot), image, K)  # БЕЗ label - как на тесте
        feat = F.normalize(h - baseline, dim=0)
        onehot = torch.zeros(10)
        onehot[label] = 1.0
        organism.write_fact_modal("mnist_readout", feat, onehot, tag_strength=1.0)
    print(f"Linear-probe обучение завершено: {n_train} цифр ({time.time()-t0:.1f}s)")

    correct = 0
    for i in test_idx:
        image, label = te_x[i], te_y[i].item()
        h = base.raw_hidden(copy.deepcopy(snapshot), image, K)
        feat = F.normalize(h - baseline, dim=0)
        pred = organism.read_fact_modal("mnist_readout", feat)
        correct += int(pred.argmax().item() == label)
    acc = correct / n_test
    print(f"ИТОГ: accuracy={acc*100:.1f}%  (N_test={n_test}, chance=10%)")
    return acc


if __name__ == "__main__":
    run()

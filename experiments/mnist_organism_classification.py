"""
Впервые в проекте: реальная задача распознавания рукописных цифр ЧЕРЕЗ
живую ткань (`unified_organism.py`), не через изолированную PredictiveCodingNet
(`mnist_pc_vs_backprop_sanity.py`, 88.4% - уже известно) и не через проверку
стабильности на одной цифре (`unified_mnist_sensory_sanity.py`).

ИСТОРИЯ (полная хронология с диагностикой - в docs/VERIFICATION_LOG.md,
"Первая попытка распознавания..." и последующие записи). Кратко, по шагам,
что было исключено, прежде чем нашлась рабочая комбинация:

1. Холст 28x28 без циклов повреждения даёт население ~13 клеток - мало.
   ИСПРАВЛЕНО: холст 64x64 + 6 циклов M7-стиля повреждение->регенерация
   (14->109 клеток).
2. Усреднённый по всем клеткам признак (hidden_representation) при малом K
   даёт вырожденный коллапс (accuracy=12%, всегда один класс). Население,
   тип признака (pooling химии/hidden_representation), нормализация,
   разнообразие bootstrap-стимула, ширина рецептивного поля (radius 1/3/5
   у ctx_kernels) - НИ ОДНО из этого само по себе не исправило коллапс.
3. НАЙДЕНА причина №1 (контрольный тест: чистые синтетические ключи дают
   100% recall - память исправна): признак РЕАЛЬНОЙ цифры почти
   неотличим от признака ПУСТОГО изображения (diff_norm на 1-2 порядка
   меньше общей нормы) - у SDR-кода (`_sdr_code`, top-k после случайной
   проекции) РАЗНЫХ цифр из-за этого 57-95% пересечения активных юнитов
   (при чистых случайных ключах пересечение было бы ~0) - Hebbian-запись
   для одной цифры катастрофически перезаписывает запись для другой.
   ИСПРАВЛЕНО: признак = (hidden_representation - baseline_на_пустом_
   изображении), НЕ сырой hidden_representation - усиливает именно
   input-специфичный остаток, пересечение SDR упало с ~0.77 до ~0.28.
4. НАЙДЕНА причина №2: train_genome=True в течение ВСЕГО потока (сотни
   цифр подряд) заставляет веса генома непрерывно ДРЕЙФОВАТЬ - diff_norm
   относительно ОДНОЙ фиксированной baseline растёт по ходу обучения
   (0.21 -> 0.83 за 50 цифр), т.е. точка отсчёта устаревает, признаки
   разных цифр перестают быть сравнимы друг с другом. ИСПРАВЛЕНО:
   геном ЗАМОРОЖЕН (train_genome=False) для всей classification-фазы, и
   КАЖДАЯ цифра (train и test) стартует с СВЕЖЕЙ копии одного и того же
   постоянного bootstrap-снапшота (`copy.deepcopy`) - устраняет и дрейф
   весов, и перенос состояния между последовательными цифрами.

РЕЗУЛЬТАТ на этой комбинации (N_TRAIN=80, N_TEST=40, быстрая проверка):
accuracy=30%, предсказания распределены по 9 из 10 классов (не коллапс) -
первое подтверждение, что self-supervised представление живой ткани
несёт классифицирующую информацию, при должном контроле confound'ов.
Этот файл - полный прогон (N_TRAIN/N_TEST по умолчанию ниже) той же
методологии для честного, статистически более весомого числа.

Методология - "linear probe" (стандартная в representation learning):
сама ткань выучила representation ТОЛЬКО self-supervised (zero backward,
PC-relaxation) ДО начала classification-фазы (во время роста и bootstrap),
затем ЗАМОРОЖЕНА - read-out поверх неё тоже zero-backward (delta-rule
Hebbian запись/чтение, не backprop, нигде).
"""
import sys, os, time, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from core.unified_organism import LivingTissue
from core.mnist_loader import load_mnist

SIZE = 64
DIGIT_SIDE = 28
GROWTH_STEPS = 300
BOOTSTRAP_CYCLES = 6
BOOTSTRAP_STEPS = 250
BOOTSTRAP_FRACTION = 0.3
K_STEPS_PER_DIGIT = 50
N_TRAIN = 600
N_TEST = 300
# GENOME_HIDDEN: найдено по ходу (K-свип показал K=50 уже near-optimal,
# K>100 при ЗАМОРОЖЕННЫХ весах даже ХУЖЕ - состояние сходится к общему
# аттрактору, а не расходится под конкретную цифру) - следующий, более
# сильный рычаг: сама сеть генома (48 скрытых нейронов - на порядок
# меньше 128 у эталонной изолированной PredictiveCodingNet, которая видит
# ВСЮ картинку 784px, не 48-мерный локальный контекст). 128 дало заметно
# лучшую разделимость (mean_pairwise_dist 1.31 vs 1.15, что важнее - min
# 0.84 vs 0.55), 256 не дало прироста сверху - 128 разумный компромисс.
GENOME_HIDDEN = 128
# SDR-память для read-out: default sdr_dim=512/sparsity=0.08 (k~41) - для
# N_TRAIN=600 записей это, вероятно, далеко за пределами надёжной ёмкости
# (тот же класс проблемы, что и найденная SDR-коллизия причины №1 в истории
# выше, просто на уровне "сколько РАЗНЫХ факторов' влезает", не "насколько
# они похожи") - увеличено на порядок, больше активных юнитов на запись.
SDR_DIM = 4096
SDR_SPARSITY = 0.05


def blob_signal(t, size=SIZE):
    s = torch.zeros(1, 2, size, size)
    c = size // 2
    r = size // 4
    s[0, 0, c - r:c + r, c - r:c + r] = 0.5
    return s


def digit_signal(image, size=SIZE, digit_side=DIGIT_SIDE):
    s = torch.zeros(1, 2, size, size)
    off = (size - digit_side) // 2
    s[0, 0, off:off + digit_side, off:off + digit_side] = image * 0.5
    return s


SPATIAL_POOL = 2  # 2x2 сетка (feat_dim=POOL*POOL*genome_hidden) - сохраняет ГДЕ
# находится штрих, не только "какой формы" в среднем; pool=1 (просто среднее)
# и pool=2 сравнимы по разделимости на малой проверке, pool=3 хуже (population
# 113 клеток на 28x28 патч - на 3x3 сетку уже не хватает покрытия по ячейкам).


def raw_hidden(org, image, k_steps, pool=SPATIAL_POOL):
    """Прогоняет K шагов PC-релаксации с ЗАМОРОЖЕННЫМИ весами генома
    (train_genome=False - см. п.4 истории выше) и возвращает пространственно
    ОБЪЕДИНЁННЫЙ (pool x pool сетка над патчем цифры, не единая точка)
    hidden_representation - сохраняет расположение штриха, не только
    усреднённую "форму в целом". growth_enabled оставлен ВКЛЮЧЁННЫМ (в
    отличие от первой рабочей версии) - т.к. каждая цифра теперь честно
    стартует со СВОЕЙ свежей копии снапшота (deepcopy), рост уже не мог бы
    "утечь" в сравнение между разными цифрами, как раньше на общей
    непрерывной ткани - проверено, не хуже (min pairwise dist чуть лучше:
    0.93 vs 0.84 при заморозке), население за K=50 шагов почти не успевает
    измениться (113->113-115), но оставлено как честный default, не
    искусственное ограничение без причины."""
    org.growth_enabled = True
    for _ in range(k_steps):
        org.step(sensory_signal=digit_signal(image), train_genome=False)
    ctx_flat, ys, xs = org.compute_context()
    h = org.hidden_representation(ctx_flat)  # (n_alive, genome_hidden)
    off = (SIZE - DIGIT_SIDE) // 2
    gh = h.shape[1]
    grid = torch.zeros(gh, SIZE, SIZE)
    grid[:, ys, xs] = h.T
    patch = grid[:, off:off + DIGIT_SIDE, off:off + DIGIT_SIDE]
    return F.adaptive_avg_pool2d(patch.unsqueeze(0), pool).flatten()


def run(seed=1):
    tr_x, tr_y, te_x, te_y = load_mnist()
    torch.manual_seed(seed)
    organism = LivingTissue(size=SIZE, state_dim=16, seed=seed, genome_hidden=GENOME_HIDDEN)

    t0 = time.time()
    for t in range(GROWTH_STEPS):
        n, err = organism.step(sensory_signal=blob_signal(t), train_genome=True)
    for cycle in range(BOOTSTRAP_CYCLES):
        killed = organism.ablate(fraction=BOOTSTRAP_FRACTION)
        for t in range(BOOTSTRAP_STEPS):
            n, err = organism.step(sensory_signal=blob_signal(t), train_genome=True)
    print(f"Популяция раскручена: {n} живых клеток ({time.time()-t0:.1f}s), геном заморожен")
    organism.growth_enabled = False

    baseline = raw_hidden(copy.deepcopy(organism), torch.zeros(DIGIT_SIDE, DIGIT_SIDE), K_STEPS_PER_DIGIT)
    print(f"Baseline (пустое изображение) вычислен, norm={baseline.norm().item():.4f}, feat_dim={baseline.numel()}")

    organism.add_modality("mnist_readout", key_dim=baseline.numel(), value_dim=10, seed=99,
                           sdr_dim=SDR_DIM, sparsity=SDR_SPARSITY)

    def feature_for_digit(image):
        org = copy.deepcopy(organism)  # СВЕЖАЯ копия снапшота - никакого переноса/дрейфа между цифрами
        h = raw_hidden(org, image, K_STEPS_PER_DIGIT)
        return F.normalize(h - baseline, dim=0)

    g = torch.Generator().manual_seed(seed + 1)
    train_idx = torch.randperm(tr_x.shape[0], generator=g)[:N_TRAIN]
    t0 = time.time()
    for count, i in enumerate(train_idx.tolist()):
        image, label = tr_x[i], tr_y[i].item()
        feat = feature_for_digit(image)
        onehot = torch.zeros(10)
        onehot[label] = 1.0
        organism.write_fact_modal("mnist_readout", feat, onehot, tag_strength=1.0)
        if (count + 1) % 100 == 0:
            print(f"  обучено {count+1}/{N_TRAIN} ({time.time()-t0:.1f}s)")
    print(f"Обучение read-out завершено: {N_TRAIN} цифр ({time.time()-t0:.1f}s)")

    test_idx = torch.randperm(te_x.shape[0], generator=torch.Generator().manual_seed(seed + 2))[:N_TEST]
    correct = 0
    per_class_correct = torch.zeros(10)
    per_class_total = torch.zeros(10)
    t0 = time.time()
    for count, i in enumerate(test_idx.tolist()):
        image, label = te_x[i], te_y[i].item()
        feat = feature_for_digit(image)
        pred = organism.read_fact_modal("mnist_readout", feat)
        pred_label = int(pred.argmax().item())
        correct += int(pred_label == label)
        per_class_total[label] += 1
        per_class_correct[label] += int(pred_label == label)
        if (count + 1) % 100 == 0:
            print(f"  протестировано {count+1}/{N_TEST} ({time.time()-t0:.1f}s)")

    acc = correct / N_TEST
    print("=" * 70)
    print(f"ИТОГ: accuracy={acc*100:.1f}%  (N_test={N_TEST}, chance=10%)")
    print("По классам:")
    for c in range(10):
        tot = int(per_class_total[c].item())
        cor = int(per_class_correct[c].item())
        pct = 100.0 * cor / tot if tot > 0 else float("nan")
        print(f"  {c}: {cor}/{tot} ({pct:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run()

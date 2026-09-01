"""
Впервые в проекте: реальная задача распознавания рукописных цифр ЧЕРЕЗ
живую ткань (`unified_organism.py`), не через изолированную PredictiveCodingNet
(`mnist_pc_vs_backprop_sanity.py`, 88.4% - уже известно) и не через проверку
стабильности на одной цифре (`unified_mnist_sensory_sanity.py`). Раньше
организм никогда не оценивался как классификатор с честным accuracy на
множестве цифр - только "не сходит ли он с ума при виде одной цифры".

Протокол:
0. ВАЖНО, найдено по ходу (rule 1 - не сдаваться после первой попытки):
   холст 28×28 (нативный размер MNIST) без циклов повреждения даёт
   население, застревающее на ~13-14 клетках (та же цифра, что и в
   пилоте M7 ДО первого цикла абляции) - слишком мало для 10-классового
   различения (первый прогон дал 12% accuracy, вырожденное решение,
   узнаёт только "1"). Популяция компаундится ЧЕРЕЗ повторные циклы
   повреждение->воспаление->регенерация (M7, docs/VERIFICATION_LOG), не
   от одного долгого роста - холст увеличен до 64×64, добавлены 6 циклов
   лёгкой (30%) абляции ДО начала классификации (14->109 клеток), цифра
   28×28 подаётся как патч в центре 64×64 поля.
1. Ткань выращивается + прогоняется через циклы бутстрэппинга населения,
   затем рост ЗАМОРАЖИВАЕТСЯ (growth_enabled=False) - иначе структура
   непредсказуемо менялась бы под потоком из сотен разных цифр, что
   сделало бы сравнение "до/после обучения" неинтерпретируемым.
2. Каждая цифра (train ИЛИ test) на несколько шагов становится сенсорным
   входом ткани (её химия релаксирует к ней через self-supervised геном -
   тот же PC-relaxation, что везде в проекте) - извлекается ОДИН вектор
   признаков (среднее hidden_representation генома по живым клеткам, тот
   же приём, что и в JEPA-мердже).
3. На train-цифрах этот вектор признаков ЗАПИСЫВАЕТСЯ в быструю
   ассоциативную SDR-память организма (add_modality/write_fact_modal,
   уже смерженный механизм) вместе с меткой (one-hot). На test-цифрах -
   ЧИТАЕТСЯ (read_fact_modal), декодируется в предсказанную метку,
   сравнивается с настоящей.

Это "linear probe"-методология (стандартная в representation learning) -
сама ткань учится ТОЛЬКО self-supervised (zero backward, как и everywhere
в проекте), read-out поверх неё - тоже zero-backward (delta-rule Hebbian
запись/чтение, не backprop). Честный вопрос: несёт ли self-supervised
представление живой ткани хоть какую-то классифицирующую информацию о
цифре - или это будет на уровне chance (10%)?
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.unified_organism import LivingTissue
from core.mnist_loader import load_mnist

SIZE = 64
DIGIT_SIDE = 28
GROWTH_STEPS = 300
BOOTSTRAP_CYCLES = 6
BOOTSTRAP_STEPS = 250
BOOTSTRAP_FRACTION = 0.3
# K_STEPS_PER_DIGIT: найдено по ходу (диагностика перед полным прогоном) -
# при k=8 признаки РАЗНЫХ цифр почти неразличимы (relative_spread=0.11,
# accuracy=12% - вырожденное "всегда 1") ПОСЛЕ bootstrap-раскрутки
# населения. Причина: геном долго обучался (2100 шагов) на ОДНОМ и том же
# generic-стимуле, химия "залипает" в этот аттрактор - k=8 шагов не хватает
# сдвинуть её под конкретную цифру. Свип k=8..400 показал: relative_spread
# растёт (0.11 -> 0.28) и НЕ выходит на плато к k=400 - k=200 - компромисс
# между разделимостью признаков и стоимостью (900 цифр * k шагов - GPU).
K_STEPS_PER_DIGIT = 200
N_TRAIN = 600
N_TEST = 300
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def organism_to_dev(o):
    o.state = o.state.to(DEV)
    o.stress_ema = o.stress_ema.to(DEV)
    o.growth_ema = o.growth_ema.to(DEV)
    o.inflammation = o.inflammation.to(DEV)
    o.ctx_kernels = o.ctx_kernels.to(DEV)
    o.dg_proj = o.dg_proj.to(DEV)
    o.W_fast = o.W_fast.to(DEV)
    g = o.genome
    g.W = [w.to(DEV) for w in g.W]
    g.b = [b.to(DEV) for b in g.b]
    if g.adam:
        g.mW = [m.to(DEV) for m in g.mW]
        g.vW = [v.to(DEV) for v in g.vW]
        g.mb = [m.to(DEV) for m in g.mb]
        g.vb = [v.to(DEV) for v in g.vb]
    if o._replay_ctx is not None:
        o._replay_ctx = o._replay_ctx.to(DEV)
        o._replay_target = o._replay_target.to(DEV)
    return o


def blob_signal(t, size=SIZE):
    s = torch.zeros(1, 2, size, size, device=DEV)
    c = size // 2
    r = size // 4
    s[0, 0, c - r:c + r, c - r:c + r] = 0.5
    return s


def digit_signal(image, size=SIZE, digit_side=DIGIT_SIDE):
    s = torch.zeros(1, 2, size, size, device=DEV)
    off = (size - digit_side) // 2
    s[0, 0, off:off + digit_side, off:off + digit_side] = image.to(DEV) * 0.5
    return s


def run(seed=1):
    tr_x, tr_y, te_x, te_y = load_mnist()
    torch.manual_seed(seed)
    organism = LivingTissue(size=SIZE, state_dim=16, seed=seed)
    organism_to_dev(organism)

    t0 = time.time()
    for t in range(GROWTH_STEPS):
        n, err = organism.step(sensory_signal=blob_signal(t), train_genome=True)
    print(f"Начальный рост: {n} живых клеток ({time.time()-t0:.1f}s)")
    for cycle in range(BOOTSTRAP_CYCLES):
        killed = organism.ablate(fraction=BOOTSTRAP_FRACTION)
        for t in range(BOOTSTRAP_STEPS):
            n, err = organism.step(sensory_signal=blob_signal(t), train_genome=True)
        print(f"  bootstrap-цикл {cycle+1}/{BOOTSTRAP_CYCLES}: killed={killed} -> {n} клеток")
    print(f"Популяция раскручена: {n} живых клеток ({time.time()-t0:.1f}s), рост заморожен")
    organism.growth_enabled = False

    # ВАЖНО, вторая находка (rule 1): усреднённый по всем клеткам
    # hidden_representation() СТИРАЕТ пространственную структуру цифры -
    # остаётся только грубая статистика типа "сколько чернил", различающая
    # разве что "1" (самая разреженная цифра) от всех остальных (тот же
    # вырожденный результат 12% - только "1" узнаётся - и при 13, и при
    # 109 клетках, population не была причиной). Вместо этого - ПРОСТРАНСТВЕННО
    # распределённый признак: химия ткани (state[2:]) внутри патча цифры,
    # усреднённая по решётке POOL×POOL, а НЕ в одну точку - сохраняет ГДЕ
    # находится штрих, не только сколько его.
    POOL = 4
    off = (SIZE - DIGIT_SIDE) // 2
    feat_dim = (organism.state_dim - 2) * POOL * POOL
    organism.add_modality("mnist_readout", key_dim=feat_dim, value_dim=10, seed=99)
    m = organism.modalities["mnist_readout"]
    m["dg_proj"] = m["dg_proj"].to(DEV)
    m["W"] = m["W"].to(DEV)

    def feature_for_digit(image):
        for _ in range(K_STEPS_PER_DIGIT):
            organism.step(sensory_signal=digit_signal(image), train_genome=True)
        chem = organism.state[0, 2:, off:off + DIGIT_SIDE, off:off + DIGIT_SIDE]
        pooled = torch.nn.functional.adaptive_avg_pool2d(chem.unsqueeze(0), POOL)
        return pooled.flatten()

    g = torch.Generator().manual_seed(seed + 1)
    train_idx = torch.randperm(tr_x.shape[0], generator=g)[:N_TRAIN]
    t0 = time.time()
    for count, i in enumerate(train_idx.tolist()):
        image, label = tr_x[i], tr_y[i].item()
        feat = feature_for_digit(image)
        onehot = torch.zeros(10, device=DEV)
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

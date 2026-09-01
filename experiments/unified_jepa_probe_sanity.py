"""
Мердж JEPA-понимания (jepa_understanding_sanity.py) в core/unified_organism.py
(приоритет (a)). Буквально скопировать текстовый пайплайн (словарь/токены/
шаблоны) в пространственный организм бессмысленно - домены разные. Вместо
этого мерджим ПРИНЦИП: скрытое представление сети, обученной ИСКЛЮЧИТЕЛЬНО
self-supervised предсказанию замаскированной части входа по остальному
(в тексте - токен по контексту; здесь - identity клетки по контексту соседей,
через compute_context()/hidden_representation(), добавленные в LivingTissue),
полезно для downstream-задачи, которой сеть НИКОГДА явно не обучалась -
и это должно бить hardcoded-правило, как и в текстовом тесте.

Downstream-задача (никогда не была целью генома): классификация локального
"режима" клетки - core (внутренняя, стабильная) vs boundary (граница ткани,
где предсказание контекста структурно труднее - сосед мёртв/неоднороден).
Метка - ground truth из топологии (есть ли мёртвый сосед), НЕ то, чему
учится геном (геном учится предсказывать identity/chemistry, не эту метку).

ЧЕСТНЫЙ HELD-OUT протокол (аналог "невиданного шаблона" из текстового теста):
1. Вырастить ткань (300 шагов) - популяция A (до повреждения).
2. Обучить ЛИНЕЙНЫЙ пробник на МАЛОЙ доле (30%) клеток популяции A -> label.
3. Откалибровать hardcoded-порог (по error_norm) на ТОЙ ЖЕ доле A.
4. Абляция (30%) + 250 шагов восстановления -> популяция B (НОВЫЙ режим:
   восстановившиеся клетки, воспалённая кайма раны - пробник и порог НЕ
   переобучаются и не перекалибровываются).
5. ЧЕСТНЫЙ ТЕСТ: применить уже обученные пробник и порог к популяции B
   (полностью held-out распределение, никогда не виденное ни геномом для
   этой метки, ни пробником/порогом при калибровке).
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.unified_organism import LivingTissue
from core.predictive_coding import PredictiveCodingNet

torch.manual_seed(42)


def signal(t, size):
    s = torch.zeros(1, 2, size, size)
    c = size // 2
    r = size // 4
    s[0, 0, c - r:c + r, c - r:c + r] = 0.5
    return s


def boundary_labels(organism, ys, xs):
    """Ground truth: 1 = граница (хотя бы один мёртвый сосед в 3x3), 0 = внутренняя."""
    alive, _ = organism.alive_mask()
    neighbor_alive_count = F.avg_pool2d(alive.float(), 3, stride=1, padding=1) * 9.0
    return (neighbor_alive_count[0, 0, ys, xs] < 8.5).long()  # <9 живых соседей включая себя -> граница


def run(size=24, grow_steps=300, seed=1):
    SIZE = size
    organism = LivingTissue(size=SIZE, state_dim=16, seed=seed)

    print("=" * 70)
    print(f"Рост популяции A ({grow_steps} шагов, SIZE={SIZE})...")
    for t in range(grow_steps):
        organism.step(sensory_signal=signal(t, SIZE), train_genome=True)
    ctx_A, ys_A, xs_A = organism.compute_context()
    hidden_A = organism.hidden_representation(ctx_A)
    labels_A = boundary_labels(organism, ys_A, xs_A)
    n_A = labels_A.shape[0]
    print(f"Популяция A: {n_A} клеток, граница={int(labels_A.sum())}, внутренние={int((1-labels_A).sum())}")

    g = torch.Generator().manual_seed(5)
    perm = torch.randperm(n_A, generator=g)
    n_train = max(4, int(0.3 * n_A))
    train_idx, calib_idx = perm[:n_train], perm[n_train:]

    print(f"\nОбучение пробника на {n_train}/{n_A} клеток (30%), калибровка hardcoded-порога на тех же...")
    hidden_dim = hidden_A.shape[1]
    Y_train = torch.zeros(n_train, 2)
    Y_train[torch.arange(n_train), labels_A[train_idx]] = 1.0
    probe = PredictiveCodingNet([hidden_dim, 2], relax_steps=30, relax_lr=0.08,
                                 weight_lr=0.02, seed=3, adam=True, weight_decay=0.02)
    for step in range(150):
        probe.train_step(hidden_A[train_idx], Y_train)

    # error_norm для hardcoded-порога - реальная ошибка предсказания генома на A (train-часть)
    target_A = organism.state[0, :, ys_A, xs_A].T
    pred_A = organism.genome.forward_pass(ctx_A)
    err_A = (target_A - pred_A).norm(dim=1)
    # Порог - медиана error_norm ИМЕННО на границе train-подвыборки vs остальных (лучший разделитель на train)
    best_th, best_train_acc = None, -1
    for q in torch.linspace(0.05, 0.95, 19):
        th = torch.quantile(err_A[train_idx], q.item())
        pred_lbl = (err_A[train_idx] > th).long()
        acc = (pred_lbl == labels_A[train_idx]).float().mean().item()
        if acc > best_train_acc:
            best_train_acc, best_th = acc, th.item()
    print(f"   hardcoded-порог откалиброван: err_norm > {best_th:.4f} -> граница (train acc={best_train_acc*100:.1f}%)")

    def probe_predict(hidden):
        logits = probe.forward_pass(hidden)
        return logits.argmax(dim=1)

    print("\n" + "-" * 70)
    print("Sanity (та же A, calibration-holdout часть, IN-distribution):")
    probe_calib_acc = (probe_predict(hidden_A[calib_idx]) == labels_A[calib_idx]).float().mean().item()
    hardcoded_calib_acc = ((err_A[calib_idx] > best_th).long() == labels_A[calib_idx]).float().mean().item()
    print(f"   Пробник:   {probe_calib_acc*100:5.1f}%")
    print(f"   Hardcoded: {hardcoded_calib_acc*100:5.1f}%")

    print("\nАбляция (30%) + 250 шагов восстановления -> популяция B (HELD-OUT режим)...")
    organism.ablate(fraction=0.3)
    for t in range(250):
        organism.step(sensory_signal=signal(t, SIZE), train_genome=True)
    ctx_B, ys_B, xs_B = organism.compute_context()
    hidden_B = organism.hidden_representation(ctx_B)
    labels_B = boundary_labels(organism, ys_B, xs_B)
    target_B = organism.state[0, :, ys_B, xs_B].T
    pred_B = organism.genome.forward_pass(ctx_B)
    err_B = (target_B - pred_B).norm(dim=1)
    n_B = labels_B.shape[0]
    print(f"Популяция B: {n_B} клеток, граница={int(labels_B.sum())}, внутренние={int((1-labels_B).sum())}")

    probe_B_acc = (probe_predict(hidden_B) == labels_B).float().mean().item()
    hardcoded_B_acc = ((err_B > best_th).long() == labels_B).float().mean().item()
    chance = max(int(labels_B.sum()), int((1 - labels_B).sum())) / n_B  # majority-class baseline, не 50%

    print("=" * 70)
    print(f"ЧЕСТНЫЙ ТЕСТ на HELD-OUT популяции B (N={n_B}, пробник/порог НЕ переобучались):")
    print(f"   Пробник (JEPA-представление генома): {probe_B_acc*100:5.1f}%")
    print(f"   Hardcoded (порог по error_norm):     {hardcoded_B_acc*100:5.1f}%")
    print(f"   Majority-class baseline:             {chance*100:5.1f}%")
    print("=" * 70)

    if probe_B_acc > hardcoded_B_acc and probe_B_acc > chance:
        print("=> JEPA-представление генома переносится на held-out режим ЛУЧШЕ hardcoded-правила -")
        print("   гипотеза подтверждена в пространственном домене, N мал (один held-out прогон, 24x24).")
    else:
        print("=> Не подтвердилось на этом прогоне - нужен перебор (масштаб/seed) прежде чем делать вывод.")

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = ["Пробник\n(JEPA-представление)", "Hardcoded\n(порог error_norm)", "Majority-class"]
    vals = [probe_B_acc * 100, hardcoded_B_acc * 100, chance * 100]
    ax.bar(methods, vals, color=["darkorange", "steelblue", "gray"])
    ax.set_ylabel("Accuracy на held-out популяции B (%)")
    ax.set_title("Понимание 'граница vs внутри' из self-supervised представления vs hardcoded")
    ax.grid(True, axis="y")
    plt.tight_layout()
    path = os.path.join(plots_dir, "unified_jepa_probe_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"\nPlot saved: {path}")

    return probe_B_acc, hardcoded_B_acc, chance


if __name__ == "__main__":
    run()

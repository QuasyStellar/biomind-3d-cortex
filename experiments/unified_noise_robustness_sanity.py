"""
Устойчивость к шумному/состязательному потоку (приоритет из README - НИ РАЗУ
не тестировалась до сих пор). Идея для состязательного случая - веб-поиск
2026-09-01: "Adversarial Takeover of Neural Cellular Automata" показывает, что
NCA уязвимы к ЦЕЛЕНАПРАВЛЕННЫМ (не случайным) возмущениям сильнее, чем к
случайному шуму той же силы - стандартный протокол в этой литературе: сравнить
не "шум vs без шума", а "состязательное возмущение vs случайный шум ТОЙ ЖЕ
нормы" - иначе "состязательность" неотличима от простого шума.

Состязательное возмущение вычислено ВРУЧНУЮ (ручной backward через известную
структуру двухслойного генома - тот же стиль, что и локальное правило в
predictive_coding.py, БЕЗ autograd/.backward()) - направление, максимизирующее
ошибку предсказания генома: d(loss)/d(input) через цепное правило по явным
формулам слоёв (dims=[in,hidden,out], L=2 - жёстко под структуру генома
организма, не общий случай).

Протокол:
1. Вырастить организм (300 шагов, чистый сигнал) - общая стартовая точка.
2. Склонировать в 5 сценариев (150 доп. шагов каждый):
   - control: чистый сигнал (baseline)
   - noise (sigma=0.2, 0.5): сигнал + случайный гауссов шум
   - adversarial (eps=0.2, 0.5, ТА ЖЕ норма возмущения, что noise): сигнал +
     eps * направление максимизации ошибки генома (пересчитывается каждый шаг)
3. Сравнить: population trend, средняя ошибка генома. Гипотеза: adversarial
   вредит СИЛЬНЕЕ noise той же нормы (не просто "шум вреден вообще").
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.unified_organism import LivingTissue

torch.manual_seed(42)


def signal(t, size):
    s = torch.zeros(1, 2, size, size)
    c = size // 2
    r = size // 4
    s[0, 0, c - r:c + r, c - r:c + r] = 0.5
    return s


def adversarial_direction(genome, ctx_flat, target_flat):
    """Направление, максимизирующее ||target - pred||^2 по ctx_flat - ручной
    backward через L=2 слоя генома (act=tanh скрытый, линейный выход), БЕЗ
    autograd. Нормировано по строкам (на клетку) до unit L2, чтобы eps ниже
    задавал ТОЧНУЮ норму возмущения, сравнимую с sigma случайного шума."""
    assert genome.L == 2, "формула ниже жёстко под двухслойный геном (as used in LivingTissue)"
    z1 = ctx_flat @ genome.W[0].T + genome.b[0]
    h1 = torch.tanh(z1)
    pred = h1 @ genome.W[1].T + genome.b[1]
    d_pred = pred - target_flat            # dL/dpred, L = ||pred-target||^2 (растёт при +err направлении)
    d_h1 = d_pred @ genome.W[1]             # dL/dh1
    d_z1 = d_h1 * (1.0 - h1 * h1)           # tanh'(z1)
    d_ctx = d_z1 @ genome.W[0]              # dL/dctx_flat - направление РОСТА ошибки
    return d_ctx / (d_ctx.norm(dim=1, keepdim=True) + 1e-7)


def run_scenario(base_organism, size, steps, mode, magnitude, seed):
    organism = base_organism.clone()
    g = torch.Generator().manual_seed(seed)
    errors, counts = [], []
    for t in range(steps):
        sig = signal(t, size)
        if mode == "noise" and magnitude > 0:
            sig = sig + magnitude * torch.randn(sig.shape, generator=g)
        elif mode == "adversarial" and magnitude > 0:
            # Возмущение вычисляется по ТЕКУЩЕМУ контексту организма ПЕРЕД шагом -
            # т.е. атака "в реальном времени", не заранее заготовленная (честнее:
            # атакующий не знает будущего состояния, только текущее).
            ctx_flat, ys, xs = organism.compute_context()
            if ctx_flat.shape[0] > 0:
                target_flat = organism.state[0, :, ys, xs].T
                adv_dir = adversarial_direction(organism.genome, ctx_flat, target_flat)
                # Возмущение задано в пространстве КОНТЕКСТА (3*state_dim), а сигнал -
                # в пространстве STATE (первые 2 канала) - проецируем через ту же долю
                # первых 2*state_dim компонент контекста (identity-часть sobel/lap
                # смешивает все каналы, поэтому берём среднее направление по alive-клеткам
                # и заливаем его в sensory-каналы 0:2, аналогично signal()).
                adv_signal_flat = adv_dir[:, :2].mean(dim=0)  # огрублённая проекция на 2 канала сигнала
                sig = sig.clone()
                sig[0, :, ys, xs] += magnitude * adv_signal_flat.unsqueeze(1)
        n, err = organism.step(sensory_signal=sig, train_genome=True)
        errors.append(err)
        counts.append(n)
    return errors, counts


def run(seed=1, scenario_seed=7):
    SIZE = 48
    print("=" * 70)
    print(f"Рост базового организма (300 шагов, чистый сигнал, seed={seed})...")
    base = LivingTissue(size=SIZE, state_dim=16, seed=seed)
    for t in range(300):
        base.step(sensory_signal=signal(t, SIZE), train_genome=True)
    base_count = int((base.state[0, 0] > 0.1).sum().item())
    print(f"Стартовая популяция для всех сценариев: {base_count} клеток")

    STEPS = 150
    scenarios = [("control", "control", 0.0)]
    for mag in [0.2, 0.5]:
        scenarios.append((f"noise sigma={mag}", "noise", mag))
        scenarios.append((f"adversarial eps={mag}", "adversarial", mag))

    results = {}
    print("\nСценарии (150 доп. шагов каждый, из ОДНОЙ стартовой точки):")
    for label, mode, mag in scenarios:
        errors, counts = run_scenario(base, SIZE, STEPS, mode, mag, seed=scenario_seed)
        results[label] = (errors, counts)
        mean_err_last50 = sum(errors[-50:]) / 50
        print(f"   {label:22s}: популяция {counts[0]:3d}->{counts[-1]:3d}   "
              f"средняя ошибка генома (посл. 50 шагов) = {mean_err_last50:.4f}")

    print("=" * 70)
    print("Честное сравнение при РАВНОЙ норме возмущения (noise vs adversarial):")
    for mag in [0.2, 0.5]:
        err_noise = sum(results[f"noise sigma={mag}"][0][-50:]) / 50
        err_adv = sum(results[f"adversarial eps={mag}"][0][-50:]) / 50
        cnt_noise = results[f"noise sigma={mag}"][1][-1]
        cnt_adv = results[f"adversarial eps={mag}"][1][-1]
        worse = "adversarial ХУЖЕ noise (как предсказывает литература)" if err_adv > err_noise else "noise НЕ хуже adversarial - гипотеза не подтвердилась на этой magnitude"
        print(f"   magnitude={mag}: err(noise)={err_noise:.4f} err(adv)={err_adv:.4f} "
              f"pop(noise)={cnt_noise} pop(adv)={cnt_adv}  => {worse}")
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, (errors, counts) in results.items():
        axes[0].plot(errors, label=label)
        axes[1].plot(counts, label=label)
    axes[0].set_title("Ошибка предсказания генома под возмущением")
    axes[0].set_xlabel("Шаг")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)
    axes[1].set_title("Популяция под возмущением")
    axes[1].set_xlabel("Шаг")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)
    plt.tight_layout()
    path = os.path.join(plots_dir, "unified_noise_robustness_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")

    return results


if __name__ == "__main__":
    run()

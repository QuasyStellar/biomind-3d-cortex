"""
Проверка с нуля: реально ли растущая/самовосстанавливающаяся ткань лучше
переживает структурное повреждение, чем та же ткань с выключенным ростом?

Это изолированный тест СТРУКТУРНОЙ динамики (не памяти/обучения — геном
здесь не обучается, см. docstring growing_tissue.py). Вопрос: даёт ли
сам механизм роста+апоптоза измеримое преимущество в восстановлении
количества/площади живых клеток после повреждения, при идентичных
условиях, кроме одного вкл/выкл флага.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.growing_tissue import GrowingTissue

torch.manual_seed(42)


def run_phase(tissue, steps, signal_fn):
    counts = []
    for t in range(steps):
        n = tissue.step(sensory_signal=signal_fn(t))
        counts.append(n)
    return counts


def run():
    GROW_STEPS = 150
    POST_ABLATION_STEPS = 150
    ABLATE_FRACTION = 0.4

    def signal(t):
        # Ограниченная зона стимула вокруг seed (не вся сетка, но и не оторвана
        # от seed - иначе расти некуда). Оставляет кольцо вне зоны без стимула,
        # где апоптоз может реально сработать - иначе рост либо не начнётся
        # (зона оторвана от seed), либо зальёт всё (зона = вся сетка).
        s = torch.zeros(1, 2, 24, 24)
        s[0, 0, 6:18, 6:18] = 0.5
        return s

    # 1. Рост из seed до стабилизации
    tissue = GrowingTissue(size=24, state_dim=16, seed=1, growth_enabled=True)
    grow_counts = run_phase(tissue, GROW_STEPS, signal)
    pre_ablation_count = grow_counts[-1]
    print(f"Рост из seed (4 клетки) -> стабилизация: {pre_ablation_count} живых клеток за {GROW_STEPS} шагов")

    # 2. Повреждение: клонируем состояние, чтобы честно сравнить growth-on vs growth-off
    #    из ОДНОЙ И ТОЙ ЖЕ точки после повреждения.
    tissue_grow_on = tissue.clone()
    tissue_grow_off = tissue.clone()
    tissue_grow_off.growth_enabled = False

    killed_on = tissue_grow_on.ablate(fraction=ABLATE_FRACTION)
    killed_off = tissue_grow_off.ablate(fraction=ABLATE_FRACTION)
    assert killed_on == killed_off, "Повреждение должно быть идентичным для честного сравнения"
    print(f"Абляция: убито {killed_on} клеток ({ABLATE_FRACTION*100:.0f}% целевой доли)")

    alive_after_ablation = int((tissue_grow_on.state[0, 0] > 0.1).sum().item())
    print(f"Живых клеток сразу после абляции: {alive_after_ablation} (было {pre_ablation_count})")

    # 3. Восстановление: рост включён vs выключен, идентичная стимуляция
    recov_on = run_phase(tissue_grow_on, POST_ABLATION_STEPS, signal)
    recov_off = run_phase(tissue_grow_off, POST_ABLATION_STEPS, signal)

    final_on = recov_on[-1]
    final_off = recov_off[-1]
    recovery_pct_on = 100.0 * final_on / pre_ablation_count
    recovery_pct_off = 100.0 * final_off / pre_ablation_count

    print("=" * 70)
    print(f"После {POST_ABLATION_STEPS} шагов восстановления:")
    print(f"  Рост ВКЛЮЧЁН:  {final_on:3d} клеток ({recovery_pct_on:5.1f}% от до-абляционного числа)")
    print(f"  Рост ВЫКЛЮЧЕН: {final_off:3d} клеток ({recovery_pct_off:5.1f}% от до-абляционного числа)")
    print(f"  Разница (регенерационное преимущество роста): {recovery_pct_on - recovery_pct_off:+.1f} п.п.")
    print("=" * 70)

    # Plot
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots"), exist_ok=True)
    plot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots", "growth_ablation_sanity.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    full_on = grow_counts + recov_on
    full_off = grow_counts[:1] * 0 + [pre_ablation_count - killed_off] + recov_off  # visual align
    ax.plot(range(len(grow_counts)), grow_counts, color="gray", linewidth=2, label="Рост из seed")
    x_post = range(GROW_STEPS, GROW_STEPS + POST_ABLATION_STEPS)
    ax.plot(x_post, recov_on, color="green", linewidth=2, label=f"Восстановление, рост ВКЛ ({recovery_pct_on:.0f}%)")
    ax.plot(x_post, recov_off, color="red", linewidth=2, linestyle="--", label=f"Восстановление, рост ВЫКЛ ({recovery_pct_off:.0f}%)")
    ax.axvline(GROW_STEPS, color="black", linestyle=":", alpha=0.6, label=f"Абляция {ABLATE_FRACTION*100:.0f}%")
    ax.axhline(pre_ablation_count, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Живых клеток")
    ax.set_title("Рост из seed -> абляция -> восстановление: рост включён vs выключен")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")

    return recovery_pct_on, recovery_pct_off, pre_ablation_count, killed_on


if __name__ == "__main__":
    run()

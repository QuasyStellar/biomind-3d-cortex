"""
BCM-подобная самонастройка порогов (M1 из ROADMAP.md): фиксированные пороги
роста/апоптоза, откалиброванные под одну интенсивность стимула, либо не
реагируют вообще при слабом стимуле, либо разрастаются неограниченно при
сильном. Самонастраивающиеся (EMA собственной активности клетки) должны
сами откалиброваться под любую интенсивность и прийти к похожему,
разумному размеру ткани в обоих режимах.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.growing_tissue import GrowingTissue

torch.manual_seed(42)


def signal_with_magnitude(mag):
    def signal(t):
        s = torch.zeros(1, 2, 24, 24)
        s[0, 0, 6:18, 6:18] = mag
        return s
    return signal


def run_condition(magnitude, homeostatic, steps=150):
    tissue = GrowingTissue(size=24, state_dim=16, seed=1, growth_enabled=True,
                            homeostatic=homeostatic,
                            neurogenesis_threshold=0.5, apoptosis_threshold=0.05)
    sig = signal_with_magnitude(magnitude)
    counts = []
    for t in range(steps):
        n = tissue.step(sensory_signal=sig(t))
        counts.append(n)
    return counts


def run():
    MAGNITUDES = {"очень слабый (0.01)": 0.01, "слабый (0.05)": 0.05, "средний (0.5)": 0.5,
                  "сильный (5.0)": 5.0, "очень сильный (50.0)": 50.0}
    results_fixed, results_homeo = {}, {}

    print("=" * 70)
    print("Фиксированные пороги (откалиброваны под 'средний' режим 0.5):")
    for name, mag in MAGNITUDES.items():
        counts = run_condition(mag, homeostatic=False)
        results_fixed[name] = counts
        print(f"  {name:18s} -> финальный размер: {counts[-1]:4d} клеток "
              f"(из 576 макс, {'ЗАЛИЛО ВСЁ' if counts[-1] > 500 else ('НЕ ВЫРОСЛО' if counts[-1] < 15 else 'норм')})")

    print("\nСамонастраивающиеся пороги (BCM-подобная EMA):")
    for name, mag in MAGNITUDES.items():
        counts = run_condition(mag, homeostatic=True)
        results_homeo[name] = counts
        print(f"  {name:18s} -> финальный размер: {counts[-1]:4d} клеток")
    print("=" * 70)

    sizes_fixed = [results_fixed[k][-1] for k in MAGNITUDES]
    sizes_homeo = [results_homeo[k][-1] for k in MAGNITUDES]
    spread_fixed = max(sizes_fixed) / max(1, min(sizes_fixed))
    spread_homeo = max(sizes_homeo) / max(1, min(sizes_homeo))
    print(f"Разброс размеров между режимами: fixed={spread_fixed:.1f}x  homeostatic={spread_homeo:.1f}x")
    print("(меньше разброс = лучше самонастройка - размер ткани не должен зависеть от абсолютной интенсивности)")

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in MAGNITUDES:
        axes[0].plot(results_fixed[name], label=name, linewidth=2)
        axes[1].plot(results_homeo[name], label=name, linewidth=2)
    axes[0].set_title(f"Фиксированные пороги (разброс {spread_fixed:.1f}x)")
    axes[1].set_title(f"Самонастраивающиеся пороги, BCM-EMA (разброс {spread_homeo:.1f}x)")
    for ax in axes:
        ax.set_xlabel("Шаг")
        ax.set_ylabel("Живых клеток")
        ax.legend()
        ax.grid(True)
        ax.set_ylim(0, 600)
    plt.tight_layout()
    path = os.path.join(plots_dir, "homeostatic_threshold_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")

    return results_fixed, results_homeo


if __name__ == "__main__":
    run()

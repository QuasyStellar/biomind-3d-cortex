"""
Честный A/B: 2D-ткань с многоканальным состоянием (`core/unified_organism.py`)
против настоящей 3D-воксельной ткани (`core/tissue_3d.py`, реализована с
нуля для этого теста) - CLAUDE_STRATEGIC_SPEC.md, раздел 1.4.

Сопоставимость масштаба: 2D size=24 (24*24=576 клеток максимум), 3D size=8
(8*8*8=512 воксилей максимум) - близкий порядок по числу вычислительных
единиц, не по стороне решётки (иначе 3D тривиально получил бы на порядки
больше юнитов). Геном - тот же PredictiveCodingNet класс в обоих случаях,
genome_hidden=48 в обоих - ЧЕСТНАЯ ОГОВОРКА: входная размерность генома
РАЗНАЯ (2D: state_dim*3=48, 3D: state_dim*4=64, т.к. 3D-перцепция
принципиально требует на одно ядро больше - Sobel по третьей оси) - не
идеально равные параметры/FLOPs, как требует спецификация, честно
отмечено, не скрыто, дальнейшая точная балансировка - не сделана в этом
первом прогоне.

Три метрики (по протоколу спецификации):
(а) скорость роста/релаксации - population и ошибка предсказания генома
    по шагам роста.
(б) ёмкость аттракторов - косвенно через то, насколько СТАБИЛЬНО
    население и ошибка держатся к концу роста (не через явную атракторную
    метрику - не реализовано отдельно, следующий шаг).
(в) устойчивость к повреждению - M7-протокол (абляция 30%, 250 шагов
    восстановления), сравнение recovery % между 2D и 3D.
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.unified_organism import LivingTissue
from core.tissue_3d import LivingTissue3D

GROWTH_STEPS = 300
RECOVERY_STEPS = 250
ABLATE_FRACTION = 0.3


def signal_2d(t, size=24):
    s = torch.zeros(1, 2, size, size)
    c = size // 2
    r = size // 4
    s[0, 0, c - r:c + r, c - r:c + r] = 0.5
    return s


def signal_3d(t, size=8):
    s = torch.zeros(1, 2, size, size, size)
    c = size // 2
    r = max(1, size // 4)
    s[0, 0, c - r:c + r, c - r:c + r, c - r:c + r] = 0.5
    return s


def run_2d(seed=1):
    torch.manual_seed(seed)
    o = LivingTissue(size=24, state_dim=16, seed=seed)
    counts, errs = [], []
    t0 = time.time()
    for t in range(GROWTH_STEPS):
        n, err = o.step(sensory_signal=signal_2d(t), train_genome=True)
        counts.append(n)
        errs.append(err)
    growth_time = time.time() - t0

    pre = counts[-1]
    killed = o.ablate(fraction=ABLATE_FRACTION)
    t0 = time.time()
    for t in range(RECOVERY_STEPS):
        n, err = o.step(sensory_signal=signal_2d(t), train_genome=True)
    recovery_time = time.time() - t0
    recovery_pct = 100.0 * n / pre if pre > 0 else 0.0
    return dict(final_pop=counts[-1], final_err=errs[-1], growth_time=growth_time,
                pre=pre, killed=killed, recovered=n, recovery_pct=recovery_pct,
                recovery_time=recovery_time, steps_per_sec=GROWTH_STEPS / growth_time)


def run_3d(seed=1):
    torch.manual_seed(seed)
    o = LivingTissue3D(size=8, state_dim=16, seed=seed)
    counts, errs = [], []
    t0 = time.time()
    for t in range(GROWTH_STEPS):
        n, err = o.step(sensory_signal=signal_3d(t), train_genome=True)
        counts.append(n)
        errs.append(err)
    growth_time = time.time() - t0

    pre = counts[-1]
    killed = o.ablate(fraction=ABLATE_FRACTION)
    t0 = time.time()
    for t in range(RECOVERY_STEPS):
        n, err = o.step(sensory_signal=signal_3d(t), train_genome=True)
    recovery_time = time.time() - t0
    recovery_pct = 100.0 * n / pre if pre > 0 else 0.0
    return dict(final_pop=counts[-1], final_err=errs[-1], growth_time=growth_time,
                pre=pre, killed=killed, recovered=n, recovery_pct=recovery_pct,
                recovery_time=recovery_time, steps_per_sec=GROWTH_STEPS / growth_time)


def run(n_seeds=3):
    results_2d, results_3d = [], []
    for seed in range(1, n_seeds + 1):
        r2 = run_2d(seed)
        r3 = run_3d(seed)
        results_2d.append(r2)
        results_3d.append(r3)
        print(f"seed={seed}")
        print(f"  2D: pop={r2['final_pop']:4d}  err={r2['final_err']:.4f}  "
              f"speed={r2['steps_per_sec']:.1f} steps/s  "
              f"recovery={r2['recovered']}/{r2['pre']} ({r2['recovery_pct']:.1f}%)")
        print(f"  3D: pop={r3['final_pop']:4d}  err={r3['final_err']:.4f}  "
              f"speed={r3['steps_per_sec']:.1f} steps/s  "
              f"recovery={r3['recovered']}/{r3['pre']} ({r3['recovery_pct']:.1f}%)")

    import statistics as st
    print("\n" + "=" * 70)
    for label, results in [("2D", results_2d), ("3D", results_3d)]:
        pops = [r["final_pop"] for r in results]
        errs = [r["final_err"] for r in results]
        recs = [r["recovery_pct"] for r in results]
        speeds = [r["steps_per_sec"] for r in results]
        print(f"{label}: pop={st.mean(pops):.1f}±{st.stdev(pops) if len(pops)>1 else 0:.1f}  "
              f"err={st.mean(errs):.4f}±{st.stdev(errs) if len(errs)>1 else 0:.4f}  "
              f"recovery%={st.mean(recs):.1f}±{st.stdev(recs) if len(recs)>1 else 0:.1f}  "
              f"speed={st.mean(speeds):.1f} steps/s")
    print("=" * 70)


if __name__ == "__main__":
    run()

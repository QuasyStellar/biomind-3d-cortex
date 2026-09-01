"""
Продолжение Synaptic Intelligence (synaptic_intelligence_forgetting_sanity.py
показал на изолированном PredictiveCodingNet чистую, монотонную кривую
компромисса; здесь - на самом организме, unified_organism.py, флагнутый
next-step). Протокол A/B/A: вырастить организм под стимулом A (сигнал в
одном месте) - геном учится предсказывать identity под статистику A;
переключить на стимул B (СОВСЕМ другое место/паттерн) - геном должен
переучиться под B; переключить ОБРАТНО на A - забыл ли геном, как хорошо
предсказывать под A, пока учился B? Сравниваем baseline (без SI) и
SI-enabled геном, идентичный протокол.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.unified_organism import LivingTissue

torch.manual_seed(42)

SIZE = 32
STEPS_PER_PHASE = 150


def signal_A(t, size):
    s = torch.zeros(1, 2, size, size)
    c = size // 2
    r = size // 5
    s[0, 0, c - r:c + r, c - r:c + r] = 0.5
    return s


def signal_B(t, size):
    # РАЗНЫЙ паттерн - кольцо вместо квадрата, другая амплитуда/частота
    s = torch.zeros(1, 2, size, size)
    c = size // 2
    val = 0.3 + 0.2 * torch.sin(torch.tensor(t * 0.3)).item()
    r_out, r_in = size // 4, size // 6
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    dist = ((yy - c) ** 2 + (xx - c) ** 2) ** 0.5
    ring = (dist > r_in) & (dist < r_out)
    s[0, 1][ring] = val
    return s


def run_phase(organism, signal_fn, steps):
    errs = []
    for t in range(steps):
        n, err = organism.step(sensory_signal=signal_fn(t, SIZE), train_genome=True)
        errs.append(err)
    return sum(errs[-30:]) / 30


def run(si_enabled, si_lambda, seed=1):
    organism = LivingTissue(size=SIZE, state_dim=16, seed=seed, si_enabled=si_enabled, si_lambda=si_lambda)

    err_A1 = run_phase(organism, signal_A, STEPS_PER_PHASE)
    if si_enabled:
        organism.genome.si_new_task()
    err_B = run_phase(organism, signal_B, STEPS_PER_PHASE)
    if si_enabled:
        organism.genome.si_new_task()
    err_A2 = run_phase(organism, signal_A, STEPS_PER_PHASE)

    return err_A1, err_B, err_A2


def run_all():
    print("=" * 70)
    print("A/B/A протокол на unified_organism.py: baseline vs Synaptic Intelligence")
    for si_enabled, si_lambda, label in [(False, 0.0, "baseline"), (True, 1.0, "SI (lambda=1.0)"), (True, 5.0, "SI (lambda=5.0)")]:
        print(f"\n--- {label} ---")
        errs_A1, errs_B, errs_A2 = [], [], []
        for seed in [1, 2, 3]:
            a1, b, a2 = run(si_enabled, si_lambda, seed=seed)
            errs_A1.append(a1); errs_B.append(b); errs_A2.append(a2)
            print(f"  seed={seed}: err(A, до B)={a1:.4f}  err(B)={b:.4f}  err(A, после B)={a2:.4f}  "
                  f"деградация={a2-a1:+.4f}")
        mean_a1 = sum(errs_A1)/3; mean_a2 = sum(errs_A2)/3; mean_b = sum(errs_B)/3
        print(f"  СРЕДНЕЕ: err(A до)={mean_a1:.4f}  err(B)={mean_b:.4f}  err(A после)={mean_a2:.4f}  "
              f"деградация={mean_a2-mean_a1:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_all()

"""
Продолжение unified_noise_robustness_sanity.py (приоритет c) - первая версия
состязательной атаки пересчитывала направление С НУЛЯ каждый шаг, без памяти
между шагами. Флагнутое, но не проверенное направление из VERIFICATION_LOG:
атака с НАКОПЛЕНИЕМ возмущения по шагам (momentum) - стандартный приём в
adversarial ML литературе (iterative/momentum FGSM сильнее single-step FGSM),
никогда раньше не пробовался в этом проекте. Реализовано с нуля: momentum-
буфер того же пространственного размера, что и сенсорный сигнал, экспоненциально
усредняющий направление атаки по позициям живых клеток кадр за кадром.

Учтён урок этой сессии (evolved-Hebbian, wiring-cost): дисперсия по seed
может быть огромной - сразу multi-seed (4), не одиночный прогон.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.unified_organism import LivingTissue
from experiments.unified_noise_robustness_sanity import signal, adversarial_direction

torch.manual_seed(42)

SIZE = 48
STEPS = 150


def run_condition(base_organism, mode, magnitude, momentum_decay, seed):
    organism = base_organism.clone()
    g = torch.Generator().manual_seed(seed)
    momentum = torch.zeros(1, 2, SIZE, SIZE)
    errors, counts = [], []
    for t in range(STEPS):
        sig = signal(t, SIZE)
        if mode == "noise" and magnitude > 0:
            sig = sig + magnitude * torch.randn(sig.shape, generator=g)
        elif mode in ("adversarial_single", "adversarial_momentum") and magnitude > 0:
            ctx_flat, ys, xs = organism.compute_context()
            if ctx_flat.shape[0] > 0:
                target_flat = organism.state[0, :, ys, xs].T
                adv_dir = adversarial_direction(organism.genome, ctx_flat, target_flat)
                adv_signal_flat = adv_dir[:, :2].mean(dim=0)
                if mode == "adversarial_momentum":
                    cur = torch.zeros(1, 2, SIZE, SIZE)
                    cur[0, :, ys, xs] = adv_signal_flat.unsqueeze(1)
                    momentum = momentum_decay * momentum + (1 - momentum_decay) * cur
                    sig = sig.clone()
                    sig += magnitude * momentum
                else:
                    sig = sig.clone()
                    sig[0, :, ys, xs] += magnitude * adv_signal_flat.unsqueeze(1)
        n, err = organism.step(sensory_signal=sig, train_genome=True)
        errors.append(err)
        counts.append(n)
    return errors, counts


def run():
    print("=" * 70)
    print(f"Momentum-накопленная атака vs single-step vs noise (4 seed, magnitude=0.3/0.5)")
    for magnitude in [0.3, 0.5]:
        results = {"noise": [], "adversarial_single": [], "adversarial_momentum": []}
        for seed in [1, 2, 3, 4]:
            base = LivingTissue(size=SIZE, state_dim=16, seed=seed)
            for t in range(300):
                base.step(sensory_signal=signal(t, SIZE), train_genome=True)
            for mode in results:
                errors, counts = run_condition(base, mode, magnitude, momentum_decay=0.7, seed=seed + 100)
                results[mode].append(sum(errors[-50:]) / 50)
        print(f"\n--- magnitude={magnitude} ---")
        for mode, vals in results.items():
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            print(f"  {mode:22s}: {[round(v,3) for v in vals]}  mean={mean:.3f}±{std:.3f}")

        noise_mean = sum(results["noise"]) / 4
        single_mean = sum(results["adversarial_single"]) / 4
        mom_mean = sum(results["adversarial_momentum"]) / 4
        print(f"  Момент vs noise: {'ХУЖЕ (сильнее атака)' if mom_mean > noise_mean else 'не хуже'} "
              f"({mom_mean:.3f} vs {noise_mean:.3f})")
        print(f"  Момент vs single-step: {'СИЛЬНЕЕ' if mom_mean > single_mean else 'не сильнее'} "
              f"({mom_mean:.3f} vs {single_mean:.3f})")
    print("=" * 70)


if __name__ == "__main__":
    run()

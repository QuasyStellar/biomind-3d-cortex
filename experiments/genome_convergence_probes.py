"""
Несколько гипотез о том, почему геном не сходится даже при стабильной
популяции клеток (1.2-1.4, без тренда вниз). Рост ВЫКЛЮЧЕН полностью
во всех прогонах - чистая изоляция вопроса "может ли геном вообще
научиться предсказывать" от вопроса роста.

Гипотезы:
A) baseline (relax_steps=15, полный target 16-dim, сенсорная инъекция каждый шаг)
B) relax_steps=45 (наши провалидированные плоские тесты использовали 40-60)
C) без сенсорной инъекции (изолировать, не она ли источник шумной цели)
D) target - только chemistry-каналы (2:), без alive/stress "бухгалтерских" каналов
E) B+D вместе
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.unified_organism import LivingTissue

torch.manual_seed(42)


def signal(t):
    s = torch.zeros(1, 2, 24, 24)
    s[0, 0, 6:18, 6:18] = 0.5
    return s


def run_variant(name, relax_steps=15, use_signal=True, chem_only_target=False, steps=250, seed=1):
    organism = LivingTissue(size=24, state_dim=16, seed=seed, growth_enabled=False,
                             relax_steps=relax_steps, predict_chem_only=chem_only_target)
    errors = []
    for t in range(steps):
        sig = signal(t) if use_signal else None
        n, err = organism.step(sensory_signal=sig, train_genome=True)
        errors.append(err)

    early = sum(errors[10:30]) / 20
    late = sum(errors[-30:]) / 30
    trend = "СХОДИТСЯ" if late < early * 0.7 else ("плато" if abs(late - early) < early * 0.15 else "не сходится/растёт")
    print(f"{name:30s} early={early:.4f}  late={late:.4f}  ({trend})")
    return errors


if __name__ == "__main__":
    print("Все прогоны: рост ВЫКЛЮЧЕН, только вопрос сходимости генома.\n")
    run_variant("A) baseline (relax=15)", relax_steps=15)
    run_variant("B) relax_steps=45", relax_steps=45)
    run_variant("C) без сенсорной инъекции", relax_steps=15, use_signal=False)
    run_variant("D) target=только chemistry", relax_steps=15, chem_only_target=True)
    run_variant("E) B+D вместе", relax_steps=45, chem_only_target=True)

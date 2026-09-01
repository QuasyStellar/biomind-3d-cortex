"""
Продолжение noise/adversarial-robustness тестов (приоритет c) - оба прошлых
теста атаковали ошибку предсказания генома, не САМО решение о росте. Флагнутое
направление: percentile-порог роста (см. unified_organism.py) - ОТНОСИТЕЛЬНЫЙ,
не абсолютный, по конструкции устойчив к РАВНОМЕРНОМУ сдвигу (весь смысл
percentile вместо константы). Вопрос: держится ли эта устойчивость на
практике под явной атакой, и уязвим ли механизм к НЕРАВНОМЕРНОЙ (целевой)
атаке, которая пытается вызвать несоразмерный рост в ОДНОМ месте?

Два условия сравниваются с noise-контролем (та же норма):
  - uniform_attack: сигнал, равномерно повышающий ошибку ВЕЗДЕ (гипотеза:
    НЕ должен влиять на решение о росте благодаря relative percentile-порогу)
  - targeted_attack: сигнал, концентрированный в ОДНОЙ области (гипотеза:
    может вызвать несоразмерный локальный рост именно там)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.unified_organism import LivingTissue

torch.manual_seed(42)

SIZE = 48
STEPS = 300


def base_signal(t, size):
    s = torch.zeros(1, 2, size, size)
    c = size // 2
    r = size // 4
    s[0, 0, c - r:c + r, c - r:c + r] = 0.5
    return s


def run_condition(mode, magnitude, seed, target_offset=(4, 4)):
    organism = LivingTissue(size=SIZE, state_dim=16, seed=seed)
    g = torch.Generator().manual_seed(seed + 200)
    counts, growth_events = [], []
    prev_alive = set()
    for t in range(STEPS):
        sig = base_signal(t, SIZE)
        if mode == "noise" and magnitude > 0:
            sig = sig + magnitude * torch.randn(sig.shape, generator=g)
        elif mode == "uniform_attack" and magnitude > 0:
            # РАВНОМЕРНО повышаем сигнал везде на ту же величину - гипотеза:
            # percentile-порог должен быть НЕЧУВСТВИТЕЛЕН к этому.
            sig = sig + magnitude
        elif mode == "targeted_attack" and magnitude > 0:
            # Концентрируем возмущение в ОДНОЙ локальной области рядом с тканью
            c = SIZE // 2
            ty, tx = c + target_offset[0], c + target_offset[1]
            sig = sig.clone()
            y0, y1 = max(0, ty - 3), min(SIZE, ty + 3)
            x0, x1 = max(0, tx - 3), min(SIZE, tx + 3)
            sig[0, 0, y0:y1, x0:x1] += magnitude * 3.0  # локально СИЛЬНЕЕ, чем noise/uniform в среднем
        n, err = organism.step(sensory_signal=sig, train_genome=True)
        counts.append(n)
    alive = organism.state[0, 0] > 0.1
    # доля живых клеток в целевой зоне (для targeted_attack) относительно её площади
    c = SIZE // 2
    ty, tx = c + target_offset[0], c + target_offset[1]
    y0, y1 = max(0, ty - 5), min(SIZE, ty + 5)
    x0, x1 = max(0, tx - 5), min(SIZE, tx + 5)
    target_zone_alive = int(alive[y0:y1, x0:x1].sum().item())
    return counts[-1], target_zone_alive


def run():
    print("=" * 70)
    print("Атака на РЕШЕНИЕ О РОСТЕ (percentile-порог), не на ошибку генома напрямую")
    print("4 seed, magnitude=0.3")
    magnitude = 0.3
    for mode in ["noise", "uniform_attack", "targeted_attack"]:
        finals, target_zones = [], []
        for seed in [1, 2, 3, 4]:
            final_n, target_alive = run_condition(mode, magnitude, seed)
            finals.append(final_n)
            target_zones.append(target_alive)
        mean_final = sum(finals) / len(finals)
        mean_target = sum(target_zones) / len(target_zones)
        print(f"  {mode:16s}: население_финал={finals} mean={mean_final:.1f}   "
              f"клеток_в_целевой_зоне={target_zones} mean={mean_target:.1f}")
    print("=" * 70)
    print("Гипотеза: uniform_attack не должен сильно менять население (percentile-порог")
    print("самонормализуется); targeted_attack может вызвать аномальный рост именно в целевой зоне.")
    print("=" * 70)


if __name__ == "__main__":
    run()

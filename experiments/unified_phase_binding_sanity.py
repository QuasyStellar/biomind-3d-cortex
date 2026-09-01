"""
Мердж phase-binding в unified_organism.py (приоритет a, последний пункт из
"колонки/JEPA/phase-binding") - LivingTissue.init_phase()/phase_sync_step().

Открытый вопрос со standalone-теста (phase_binding_sanity.py): механизм
математически корректен, но качество сегментации на СЛУЧАЙНЫХ representations
зависит от знака cos-сходства (положительное -> сегментации нет). Реальная
химия ткани НЕ случайна - она формируется PC-релаксацией генома под РАЗНЫЕ
сенсорные входы. Вопрос: возникает ли у РЕАЛЬНО обученной химии двух разных
(но физически СОПРИКАСАЮЩИХСЯ) областей нужное для сегментации отрицательное/
околонулевое сходство САМА ПО СЕБЕ, или потребуется явный механизм
декорреляции (не проверено здесь, отдельный будущий вопрос)?

Сцена: один прямоугольный блок клеток (рост ВЫКЛЮЧЕН - тестируем чистую
химическую дифференциацию, не смешиваем со структурной динамикой), левая
половина получает СВОЙ периодический сенсорный сигнал, правая половина -
ДРУГОЙ, все N шагов обучения генома. Обе половины физически СОПРИКАСАЮТСЯ
(граница - прямые соседи), начальная химия - одинаковый случайный шум
(различие возникает ТОЛЬКО из обучения, не задано вручную).

Сравнение:
  - "trained": химия после N шагов обучения под разными сигналами
  - "untrained" (baseline): та же сцена, БЕЗ шагов обучения (химия остаётся
    случайным шумом с самого начала) - контроль, что дифференциация НЕ
    происходит просто от инициализации.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.unified_organism import LivingTissue
from core.phase_binding import order_parameter

torch.manual_seed(42)

SIZE = 32
BLOCK = (8, 24, 4, 28)


def build_organism(seed=1):
    organism = LivingTissue(size=SIZE, state_dim=16, seed=seed, growth_enabled=False)
    y0, y1, x0, x1 = BLOCK
    xmid = (x0 + x1) // 2
    g = torch.Generator().manual_seed(seed)
    organism.state.zero_()
    organism.state[0, 0, y0:y1, x0:x1] = 1.0  # alive
    organism.state[0, 2:, y0:y1, x0:x1] = 0.1 * torch.randn(organism.state_dim - 2, y1 - y0, x1 - x0, generator=g)
    mask_A = torch.zeros(SIZE, SIZE, dtype=torch.bool); mask_A[y0:y1, x0:xmid] = True
    mask_B = torch.zeros(SIZE, SIZE, dtype=torch.bool); mask_B[y0:y1, xmid:x1] = True
    return organism, mask_A, mask_B


def signal_for_half(t, mask, size, freq, amp=0.5):
    s = torch.zeros(1, 2, size, size)
    val = amp * (0.5 + 0.5 * torch.sin(torch.tensor(t * freq)))
    s[0, 0][mask] = val
    return s


def run():
    print("=" * 70)
    print("TRAINED: две половины получают РАЗНЫЕ периодические сигналы, 400 шагов обучения генома...")
    organism_t, mask_A, mask_B = build_organism(seed=1)
    for t in range(400):
        sig = signal_for_half(t, mask_A, SIZE, freq=0.2) + signal_for_half(t, mask_B, SIZE, freq=0.05)
        organism_t.step(sensory_signal=sig, train_genome=True)

    chem = organism_t.state[0, 2:]
    chem_A = chem[:, mask_A].mean(dim=1)
    chem_B = chem[:, mask_B].mean(dim=1)
    cos_sim_trained = torch.nn.functional.cosine_similarity(chem_A, chem_B, dim=0).item()
    print(f"   cos_sim(chem_A, chem_B) ПОСЛЕ обучения: {cos_sim_trained:.3f}")

    organism_t.init_phase(seed=1)
    for t in range(3000):
        organism_t.phase_sync_step(K=1.0, dt=0.1, sim_gate=True)
    r_A_t = order_parameter(organism_t.phase, mask_A)
    r_B_t = order_parameter(organism_t.phase, mask_B)
    r_all_t = order_parameter(organism_t.phase, mask_A | mask_B)
    print(f"   phase-sync: r_A={r_A_t:.3f} r_B={r_B_t:.3f} r(A∪B)={r_all_t:.3f}")

    print("\nBASELINE (untrained): та же сцена БЕЗ шагов обучения (химия = исходный шум)...")
    organism_u, mask_A2, mask_B2 = build_organism(seed=1)
    chem_u = organism_u.state[0, 2:]
    chem_A_u = chem_u[:, mask_A2].mean(dim=1)
    chem_B_u = chem_u[:, mask_B2].mean(dim=1)
    cos_sim_untrained = torch.nn.functional.cosine_similarity(chem_A_u, chem_B_u, dim=0).item()
    print(f"   cos_sim(chem_A, chem_B) БЕЗ обучения: {cos_sim_untrained:.3f}")

    organism_u.init_phase(seed=1)
    for t in range(3000):
        organism_u.phase_sync_step(K=1.0, dt=0.1, sim_gate=True)
    r_A_u = order_parameter(organism_u.phase, mask_A2)
    r_B_u = order_parameter(organism_u.phase, mask_B2)
    r_all_u = order_parameter(organism_u.phase, mask_A2 | mask_B2)
    print(f"   phase-sync: r_A={r_A_u:.3f} r_B={r_B_u:.3f} r(A∪B)={r_all_u:.3f}")

    print("=" * 70)
    print("Честный вывод:")
    print(f"   trained:   cos_sim={cos_sim_trained:.3f}  r(A∪B)={r_all_t:.3f}")
    print(f"   untrained: cos_sim={cos_sim_untrained:.3f}  r(A∪B)={r_all_u:.3f}")
    if r_all_t < r_all_u - 0.1:
        print("   => Обучение под разными сигналами РЕАЛЬНО улучшает сегментацию по сравнению со случайной химией")
    elif abs(r_all_t - r_all_u) <= 0.1:
        print("   => Обучение НЕ дало заметной разницы в сегментации - химия не задифференцировалась в нужном направлении (cos_sim знак решает всё, как и в standalone-тесте)")
    else:
        print("   => Обучение УХУДШИЛО сегментацию по сравнению со случайной химией - неожиданно, зафиксировано честно")
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["trained\ncos_sim=%.2f" % cos_sim_trained, "untrained\ncos_sim=%.2f" % cos_sim_untrained]
    vals = [r_all_t, r_all_u]
    ax.bar(labels, vals, color=["darkorange", "steelblue"])
    ax.set_ylabel("r(A∪B) - НИЗКОЕ = хорошая сегментация")
    ax.set_title("Phase-binding на РЕАЛЬНОЙ обученной химии организма")
    ax.grid(True, axis="y")
    plt.tight_layout()
    path = os.path.join(plots_dir, "unified_phase_binding_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")


if __name__ == "__main__":
    run()

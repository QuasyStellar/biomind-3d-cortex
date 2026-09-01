"""
Первая проверка phase-binding (core/phase_binding.py) - компонент никогда
раньше не реализовывался в этом проекте вообще (последний пункт из списка
"колонки, JEPA-понимание, phase-binding", остальные два уже слиты).

Классический тест binding-by-synchrony (Singer/Gray, von der Malsburg):
ДВА пространственно СОПРИКАСАЮЩИХСЯ объекта (не разделённые физически -
иначе сегментация тривиальна просто из-за отсутствия связи) с РАЗНОЙ
"химией" (представлением) должны САМООРГАНИЗОВАННО расcинхронизироваться
по фазе МЕЖДУ собой, но синхронизироваться ВНУТРИ каждого - несмотря на то,
что клетки на границе объектов - прямые физические соседи.

Сцена: один прямоугольный блок живых клеток, левая половина = "объект A"
(общая химия + шум), правая половина = "объект B" (другая химия + шум) -
физически ОДИН связный кусок, граница A/B - прямые соседи.

Сравниваем:
  - test (sim_gate=True): связь взвешена косинусным сходством химии -
    гипотеза: A и B расcинхронизируются несмотря на физическое соседство
  - baseline (sim_gate=False): связь по чистой топологии (все соседи
    равновесны) - ожидание: весь связный блок сойдётся к ОДНОЙ фазе
    (не сегментирует, т.к. это один физически связный кусок)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.phase_binding import kuramoto_step, order_parameter

torch.manual_seed(42)

SIZE = 32
CHEM_DIM = 32
BLOCK = (8, 24, 4, 28)  # y0,y1,x0,x1


def build_scene(seed=1):
    g = torch.Generator().manual_seed(seed)
    y0, y1, x0, x1 = BLOCK
    alive = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    alive[y0:y1, x0:x1] = True
    xmid = (x0 + x1) // 2
    mask_A = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    mask_A[y0:y1, x0:xmid] = True
    mask_B = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    mask_B[y0:y1, xmid:x1] = True

    identity_A = torch.nn.functional.normalize(torch.randn(CHEM_DIM, generator=g), dim=0)
    identity_B = torch.nn.functional.normalize(torch.randn(CHEM_DIM, generator=g), dim=0)
    print(f"Косинусное сходство identity_A/identity_B (должно быть низким): "
          f"{torch.nn.functional.cosine_similarity(identity_A, identity_B, dim=0).item():.3f}")

    chem = torch.zeros(CHEM_DIM, SIZE, SIZE)
    noise = 0.15 * torch.randn(CHEM_DIM, SIZE, SIZE, generator=g)
    chem[:, mask_A] = identity_A.unsqueeze(1) + noise[:, mask_A]
    chem[:, mask_B] = identity_B.unsqueeze(1) + noise[:, mask_B]

    phase = 2 * torch.pi * torch.rand(SIZE, SIZE, generator=g)
    return chem, phase, alive, mask_A, mask_B


def run_condition(sim_gate, steps=400, K=2.0, dt=0.3, seed=1):
    chem, phase, alive, mask_A, mask_B = build_scene(seed=seed)
    r_A_hist, r_B_hist, r_all_hist = [], [], []
    for t in range(steps):
        phase = kuramoto_step(chem, phase, alive, K=K, dt=dt, sim_gate=sim_gate)
        r_A_hist.append(order_parameter(phase, mask_A))
        r_B_hist.append(order_parameter(phase, mask_B))
        r_all_hist.append(order_parameter(phase, mask_A | mask_B))
    return r_A_hist, r_B_hist, r_all_hist


def run():
    print("=" * 70)
    print("Условие TEST (sim_gate=True, связь взвешена сходством химии):")
    rA_t, rB_t, rAll_t = run_condition(sim_gate=True)
    print(f"   финальный r(A)={rA_t[-1]:.3f}  r(B)={rB_t[-1]:.3f}  r(A∪B)={rAll_t[-1]:.3f}")

    print("\nУсловие BASELINE (sim_gate=False, чистая топология, без сходства):")
    rA_f, rB_f, rAll_f = run_condition(sim_gate=False)
    print(f"   финальный r(A)={rA_f[-1]:.3f}  r(B)={rB_f[-1]:.3f}  r(A∪B)={rAll_f[-1]:.3f}")

    print("=" * 70)
    print("Честная интерпретация:")
    print(f"   TEST: внутри-объектная синхронность высокая (r_A={rA_t[-1]:.3f}, r_B={rB_t[-1]:.3f}), "
          f"межобъектная НИЗКАЯ (r_A∪B={rAll_t[-1]:.3f}) -> {'СЕГМЕНТАЦИЯ ЕСТЬ' if min(rA_t[-1],rB_t[-1]) > 0.7 and rAll_t[-1] < 0.5 else 'сегментация НЕ подтверждена по этим порогам'}")
    print(f"   BASELINE: r_A∪B={rAll_f[-1]:.3f} -> {'весь связный блок слился в одну фазу (ожидаемо, контроль)' if rAll_f[-1] > 0.7 else 'baseline тоже не слился - неожиданно, требует объяснения'}")
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(rA_t, label="r(A)", color="darkorange")
    axes[0].plot(rB_t, label="r(B)", color="steelblue")
    axes[0].plot(rAll_t, label="r(A∪B)", color="black", linestyle="--")
    axes[0].set_title("TEST: связь взвешена сходством химии")
    axes[0].set_xlabel("Шаг")
    axes[0].set_ylabel("Kuramoto order parameter r")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(rA_f, label="r(A)", color="darkorange")
    axes[1].plot(rB_f, label="r(B)", color="steelblue")
    axes[1].plot(rAll_f, label="r(A∪B)", color="black", linestyle="--")
    axes[1].set_title("BASELINE: чистая топология (без сходства)")
    axes[1].set_xlabel("Шаг")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    path = os.path.join(plots_dir, "phase_binding_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")

    return rA_t, rB_t, rAll_t, rA_f, rB_f, rAll_f


if __name__ == "__main__":
    run()

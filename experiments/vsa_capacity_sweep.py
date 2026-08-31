"""
Реальный тест "The Binding Problem" с нуля: не одно число (как v1's 97.7%
на фиксированной задаче), а ёмкостная кривая — сколько одновременно
связанных role-filler пар можно суперпозировать (сложить) в один вектор
и всё ещё корректно развязать обратно через circular correlation.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.vsa_binding import circular_conv, circular_corr, random_vectors

torch.manual_seed(42)

DIM = 1024
N_ENTITIES = 200
N_ROLES = 512  # roles ~ "slot 1", "slot 2", ... (могло бы быть subj/rel/obj/time/place и т.д.)


def decode(vec, entity_vecs):
    sims = torch.nn.functional.cosine_similarity(entity_vecs, vec.unsqueeze(0), dim=-1)
    return int(sims.argmax().item())


def run_capacity_test(K_values, trials=20):
    role_vecs = random_vectors(N_ROLES, DIM, seed=1)
    entity_vecs = random_vectors(N_ENTITIES, DIM, seed=2)

    results = []
    for K in K_values:
        g = torch.Generator().manual_seed(100 + K)
        correct = 0
        total = 0
        for _ in range(trials):
            filler_idx = torch.randint(0, N_ENTITIES, (K,), generator=g)
            scene = torch.zeros(DIM)
            for k in range(K):
                scene = scene + circular_conv(role_vecs[k], entity_vecs[filler_idx[k]])
            for k in range(K):
                retrieved = circular_corr(scene, role_vecs[k])
                pred = decode(retrieved, entity_vecs)
                if pred == filler_idx[k].item():
                    correct += 1
                total += 1
        acc = correct / total
        results.append((K, acc))
        print(f"K={K:3d} одновременно связанных пар  ->  decode accuracy={acc*100:5.1f}%  (N={total})")
    return results


def run():
    K_values = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
    results = run_capacity_test(K_values, trials=20)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(*zip(*results), "b-o", linewidth=2)
    ax.axhline(1.0 / N_ENTITIES * 100, color="gray", linestyle=":", label=f"Случайный шанс ({1/N_ENTITIES*100:.1f}%)")
    ax.set_xlabel("K — число одновременно связанных role-filler пар в одном векторе")
    ax.set_ylabel("Decode accuracy (%)")
    ax.set_title(f"VSA/HRR binding capacity (dim={DIM}, {N_ENTITIES} сущностей)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(plots_dir, "vsa_capacity_sweep.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")
    return results


if __name__ == "__main__":
    run()

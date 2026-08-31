"""
Retention-кривая на реальном масштабе (N=50..4000 фактов) — v1 никогда
не проверяла дальше ~150 фактов. Сравнение SDR (разреженный код) против
плотного baseline с тем же Хеббовским правилом, при идентичных условиях —
изолирует причинный вклад именно разреженности, а не общего механизма.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.sdr_hippocampus import SDRHippocampus, DenseHippocampus

torch.manual_seed(42)

DIM = 64
N_ENTITIES = 400
N_RELATIONS = 30


def make_facts(n, seed=42):
    g = torch.Generator().manual_seed(seed)
    entity_vecs = torch.nn.functional.normalize(torch.randn(N_ENTITIES, DIM, generator=g), dim=-1)
    relation_vecs = torch.nn.functional.normalize(torch.randn(N_RELATIONS, DIM, generator=g), dim=-1)

    seen_keys = set()
    facts = []
    while len(facts) < n:
        s = torch.randint(0, N_ENTITIES, (1,), generator=g).item()
        r = torch.randint(0, N_RELATIONS, (1,), generator=g).item()
        o = torch.randint(0, N_ENTITIES, (1,), generator=g).item()
        if s == o or (s, r) in seen_keys:
            continue
        seen_keys.add((s, r))
        facts.append((s, r, o))
    return facts, entity_vecs, relation_vecs


def decode(pred_vec, entity_vecs):
    sims = entity_vecs @ pred_vec
    return int(sims.argmax().item())


def eval_retention(hippo, facts, entity_vecs, relation_vecs):
    correct = 0
    for s, r, o in facts:
        key = torch.nn.functional.normalize(entity_vecs[s] + relation_vecs[r], dim=0)
        pred = hippo.read(key)
        if decode(pred, entity_vecs) == o:
            correct += 1
    return correct / len(facts)


def run():
    N_MAX = 4000
    CHECKPOINTS = [50, 200, 500, 1000, 2000, 4000]

    all_facts, entity_vecs, relation_vecs = make_facts(N_MAX)

    sdr_hippo = SDRHippocampus(dim=DIM, sdr_dim=1024, sparsity=0.06, beta=0.9, seed=1)
    dense_hippo = DenseHippocampus(dim=DIM, beta=0.9, seed=1)

    sdr_curve, dense_curve = [], []
    for i, (s, r, o) in enumerate(all_facts, 1):
        key = torch.nn.functional.normalize(entity_vecs[s] + relation_vecs[r], dim=0)
        val = entity_vecs[o]
        sdr_hippo.write(key, val)
        dense_hippo.write(key, val)

        if i in CHECKPOINTS:
            sdr_acc = eval_retention(sdr_hippo, all_facts[:i], entity_vecs, relation_vecs)
            dense_acc = eval_retention(dense_hippo, all_facts[:i], entity_vecs, relation_vecs)
            sdr_curve.append((i, sdr_acc))
            dense_curve.append((i, dense_acc))
            print(f"N={i:5d}  SDR retention={sdr_acc*100:5.1f}%   Dense retention={dense_acc*100:5.1f}%")

    print("=" * 70)
    print(f"Финал N={N_MAX}: SDR={sdr_curve[-1][1]*100:.1f}%  Dense={dense_curve[-1][1]*100:.1f}%  "
          f"Разница={  (sdr_curve[-1][1]-dense_curve[-1][1])*100:+.1f} п.п.")
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(*zip(*sdr_curve), "g-o", label="SDR (разреженный код, k=6%)", linewidth=2)
    ax.plot(*zip(*dense_curve), "r--s", label="Dense (плотный ключ, тот же Hebbian)", linewidth=2)
    ax.set_xlabel("Число накопленных фактов (N)")
    ax.set_ylabel("Retention (recall accuracy по всем фактам)")
    ax.set_title("Retention-кривая: SDR vs Dense Hebbian память, N до 4000")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(plots_dir, "hippocampus_retention_sweep.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")
    return sdr_curve, dense_curve


if __name__ == "__main__":
    run()

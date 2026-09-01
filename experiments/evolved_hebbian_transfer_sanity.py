"""
Проверка M2 (ROADMAP.md): правило пластичности, найденное эволюцией на
задаче A, должно переноситься на задачу B (другая статистика/"модальность")
БЕЗ переподбора коэффициентов - лучше, чем правило, вручную настроенное
конкретно под A.

Задача A: обычные unit-norm entity-векторы (как в предыдущих тестах).
Задача B: другая статистика - НЕ нормированные векторы заметно большего
масштаба (имитация другой модальности с другим intrinsic scale, например
аудио-спектрограммы вместо визуальных эмбеддингов).

Сравниваем на задаче B: (1) правило эволюционировано на A и перенесено
без изменений, (2) beta вручную откалибрована конкретно под A (grid search),
перенесена без изменений, (3) правило эволюционировано сразу на B (upper bound).
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.evolved_hebbian import EvolvedHebbianHippocampus, evolve_abcd

torch.manual_seed(42)

DIM = 64
N_ENTITIES = 300
N_RELATIONS = 20
N_FACTS_FITNESS = 150   # для скорости эволюции/grid-search
N_FACTS_FINAL = 400     # для финальной честной оценки найденных правил


def make_facts(n, scale=1.0, seed=42):
    g = torch.Generator().manual_seed(seed)
    entity_vecs = torch.randn(N_ENTITIES, DIM, generator=g)
    entity_vecs = torch.nn.functional.normalize(entity_vecs, dim=-1) * scale
    relation_vecs = torch.nn.functional.normalize(torch.randn(N_RELATIONS, DIM, generator=g), dim=-1)
    seen, facts = set(), []
    while len(facts) < n:
        s = torch.randint(0, N_ENTITIES, (1,), generator=g).item()
        r = torch.randint(0, N_RELATIONS, (1,), generator=g).item()
        o = torch.randint(0, N_ENTITIES, (1,), generator=g).item()
        if s == o or (s, r) in seen:
            continue
        seen.add((s, r))
        facts.append((s, r, o))
    return facts, entity_vecs, relation_vecs


def eval_retention(coeffs, facts, entity_vecs, relation_vecs, seed=1):
    hippo = EvolvedHebbianHippocampus(dim=DIM, sdr_dim=1024, sparsity=0.06, coeffs=coeffs, seed=seed)
    for s, r, o in facts:
        key = torch.nn.functional.normalize(entity_vecs[s] + relation_vecs[r], dim=0)
        hippo.write(key, entity_vecs[o])
    correct = 0
    for s, r, o in facts:
        key = torch.nn.functional.normalize(entity_vecs[s] + relation_vecs[r], dim=0)
        pred = hippo.read(key)
        sims = entity_vecs @ pred
        if int(sims.argmax().item()) == o:
            correct += 1
    return correct / len(facts)


def run():
    facts_A, ent_A, rel_A = make_facts(N_FACTS_FITNESS, scale=1.0, seed=42)
    facts_B, ent_B, rel_B = make_facts(N_FACTS_FITNESS, scale=4.0, seed=99)  # другой масштаб = другая "модальность"

    print("=" * 70)
    print("1) Эволюция ABCD-правила на задаче A (unit-scale)...")
    best_A, fit_A, hist_A = evolve_abcd(
        lambda c: eval_retention(c, facts_A, ent_A, rel_A), generations=20, population=16, elite=4, seed=1)
    print(f"   Найдено: A={best_A[0]:.3f} B={best_A[1]:.3f} C={best_A[2]:.3f} D={best_A[3]:.3f}  fitness(A)={fit_A*100:.1f}%")

    print("2) Grid-search ручной beta, откалиброванной конкретно под задачу A...")
    best_beta, best_beta_fit = None, -1
    for beta in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0]:
        fit = eval_retention((beta, -beta, 0.0, 0.0), facts_A, ent_A, rel_A)
        if fit > best_beta_fit:
            best_beta_fit, best_beta = fit, beta
    print(f"   Найдено: beta={best_beta:.2f}  fitness(A)={best_beta_fit*100:.1f}%")

    print("3) Эволюция ABCD-правила НАПРЯМУЮ на задаче B (upper bound)...")
    best_B, fit_B, hist_B = evolve_abcd(
        lambda c: eval_retention(c, facts_B, ent_B, rel_B), generations=20, population=16, elite=4, seed=2)
    print(f"   Найдено: A={best_B[0]:.3f} B={best_B[1]:.3f} C={best_B[2]:.3f} D={best_B[3]:.3f}  fitness(B)={fit_B*100:.1f}%")

    # --- ФИНАЛЬНОЕ ЧЕСТНОЕ СРАВНЕНИЕ на задаче B, больший масштаб N ---
    facts_B_final, ent_B_final, rel_B_final = make_facts(N_FACTS_FINAL, scale=4.0, seed=99)

    acc_evolved_on_A_transferred = eval_retention(best_A, facts_B_final, ent_B_final, rel_B_final)
    acc_handtuned_on_A_transferred = eval_retention((best_beta, -best_beta, 0.0, 0.0), facts_B_final, ent_B_final, rel_B_final)
    acc_evolved_direct_on_B = eval_retention(best_B, facts_B_final, ent_B_final, rel_B_final)

    print("=" * 70)
    print(f"ФИНАЛЬНОЕ СРАВНЕНИЕ на задаче B (N={N_FACTS_FINAL}, перенос без переподбора):")
    print(f"  (1) Эволюционировано на A, перенесено на B: {acc_evolved_on_A_transferred*100:5.1f}%")
    print(f"  (2) Вручную настроено на A, перенесено на B: {acc_handtuned_on_A_transferred*100:5.1f}%")
    print(f"  (3) Эволюционировано напрямую на B (upper bound): {acc_evolved_direct_on_B*100:5.1f}%")
    print("=" * 70)

    gap_evolved = acc_evolved_direct_on_B - acc_evolved_on_A_transferred
    gap_handtuned = acc_evolved_direct_on_B - acc_handtuned_on_A_transferred
    print(f"Разрыв до upper bound: эволюция-перенос={gap_evolved*100:.1f} п.п., "
          f"ручная-перенос={gap_handtuned*100:.1f} п.п.")
    if gap_evolved < gap_handtuned:
        print("=> Эволюционированное правило переносится ЛУЧШЕ ручного - гипотеза M2 подтверждена")
    else:
        print("=> Эволюционированное правило НЕ лучше ручного при переносе - гипотеза M2 не подтвердилась")

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["(1) Эволюция A\n-> перенос на B", "(2) Ручная beta A\n-> перенос на B", "(3) Эволюция\nнапрямую на B"]
    vals = [acc_evolved_on_A_transferred*100, acc_handtuned_on_A_transferred*100, acc_evolved_direct_on_B*100]
    colors = ["darkorange", "steelblue", "green"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Retention accuracy на задаче B (%)")
    ax.set_title(f"Перенос правила пластичности между задачами разного масштаба (N={N_FACTS_FINAL})")
    ax.grid(True, axis="y")
    plt.tight_layout()
    path = os.path.join(plots_dir, "evolved_hebbian_transfer_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")

    return acc_evolved_on_A_transferred, acc_handtuned_on_A_transferred, acc_evolved_direct_on_B


if __name__ == "__main__":
    run()

"""
Продолжение M2 (evolved_hebbian_transfer_sanity.py дал ОТРИЦАТЕЛЬНЫЙ
результат: правило, эволюционированное на ОДНОЙ задаче A, переносится на
задачу B НЕ лучше вручную настроенной beta). По правилу 1 (не бросай гипотезу
после первой неудачи) и по явной просьбе пользователя ("evolved hebbian не
отбрасывай, может ещё всё впереди") - перед тем как считать M2 закрытым,
нужно попробовать стандартный в литературе подход, которого мы ЕЩЁ не
пробовали: эволюция НЕ на одной задаче, а на РАСПРЕДЕЛЕНИИ задач сразу.

Источник идеи (веб-поиск 2026-09-01, см. VERIFICATION_LOG): Najarro & Risi,
"Meta-Learning through Hebbian Plasticity in Random Networks" (тот же ABCD-
формализм, что мы уже используем) - их правила генерализуются ТОЛЬКО когда
эволюционный fitness усредняется по РАСПРЕДЕЛЕНИЮ окружений/задач, а не по
одной; на одной задаче эволюция вырождается в переобучение под её конкретную
статистику - то есть наш прошлый негативный результат предсказан литературой
как ОЖИДАЕМЫЙ провал ИМЕННО single-task протокола, а не провал самой идеи
эволюции правил вообще. (Похожий принцип независимо нашёлся в MetaNCA -
conditioning локального правила на несколько задач одновременно даёт задаче-
специфичную специализацию внутри общего параметрического пространства,
не требуя переподбора с нуля.)

Протокол (сравнение на held-out задаче B, scale=4.0, как в предыдущем тесте):
  (1) эволюция на ОДНОЙ задаче A (scale=1.0) -> перенос [уже известно: провал]
  (2) ручная beta на A -> перенос [baseline из прошлого теста]
  (3) НОВОЕ: эволюция на РАСПРЕДЕЛЕНИИ из 3 задач (scale=1.0, 2.0, 0.7,
      НИ ОДНА не равна held-out scale=4.0) - fitness = среднее retention
      по всем трём одновременно -> перенос на B БЕЗ переподбора
  (4) эволюция напрямую на B (upper bound, честный потолок)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.evolved_hebbian import evolve_abcd
from experiments.evolved_hebbian_transfer_sanity import make_facts, eval_retention

torch.manual_seed(42)

N_FACTS_FITNESS = 150
N_FACTS_FINAL = 400
TRAIN_SCALES = [1.0, 2.0, 0.7]   # распределение задач для мульти-task эволюции
HELDOUT_SCALE = 4.0              # НИ РАЗУ не виден эволюцией/подбором


def run():
    # Готовим по одному набору фактов на каждую задачу распределения (разные seed'ы,
    # чтобы задачи отличались не только масштабом, но и конкретной выборкой).
    train_tasks = [make_facts(N_FACTS_FITNESS, scale=s, seed=42 + i)
                    for i, s in enumerate(TRAIN_SCALES)]

    print("=" * 70)
    print(f"1) Эволюция ABCD на РАСПРЕДЕЛЕНИИ из {len(TRAIN_SCALES)} задач (scales={TRAIN_SCALES})...")

    def multitask_fitness(coeffs):
        accs = [eval_retention(coeffs, facts, ent, rel) for facts, ent, rel in train_tasks]
        return sum(accs) / len(accs)

    best_multi, fit_multi, hist_multi = evolve_abcd(
        multitask_fitness, generations=20, population=16, elite=4, seed=3)
    print(f"   Найдено: A={best_multi[0]:.3f} B={best_multi[1]:.3f} C={best_multi[2]:.3f} D={best_multi[3]:.3f}")
    print(f"   Средний fitness по {len(TRAIN_SCALES)} обучающим задачам: {fit_multi*100:.1f}%")

    # --- Baselines: пересчитываем single-task эволюцию и ручную beta (как в
    # прошлом тесте, тот же протокол scale=1.0 -> перенос на scale=4.0) ---
    facts_A, ent_A, rel_A = make_facts(N_FACTS_FITNESS, scale=1.0, seed=42)
    print("\n2) Baseline (перезапуск): эволюция на ОДНОЙ задаче A (scale=1.0)...")
    best_A, fit_A, _ = evolve_abcd(
        lambda c: eval_retention(c, facts_A, ent_A, rel_A), generations=20, population=16, elite=4, seed=1)
    print(f"   fitness(A)={fit_A*100:.1f}%")

    print("3) Baseline: ручная beta, grid-search на A...")
    best_beta, best_beta_fit = None, -1
    for beta in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0]:
        fit = eval_retention((beta, -beta, 0.0, 0.0), facts_A, ent_A, rel_A)
        if fit > best_beta_fit:
            best_beta_fit, best_beta = fit, beta
    print(f"   beta={best_beta:.2f}  fitness(A)={best_beta_fit*100:.1f}%")

    print(f"\n4) Upper bound: эволюция НАПРЯМУЮ на held-out задаче B (scale={HELDOUT_SCALE})...")
    facts_B, ent_B, rel_B = make_facts(N_FACTS_FITNESS, scale=HELDOUT_SCALE, seed=99)
    best_B, fit_B, _ = evolve_abcd(
        lambda c: eval_retention(c, facts_B, ent_B, rel_B), generations=20, population=16, elite=4, seed=2)
    print(f"   fitness(B)={fit_B*100:.1f}%")

    # --- ЧЕСТНОЕ ФИНАЛЬНОЕ СРАВНЕНИЕ на held-out B, больший масштаб N, БЕЗ переподбора ---
    facts_B_final, ent_B_final, rel_B_final = make_facts(N_FACTS_FINAL, scale=HELDOUT_SCALE, seed=99)

    acc_single_transferred = eval_retention(best_A, facts_B_final, ent_B_final, rel_B_final)
    acc_handtuned_transferred = eval_retention((best_beta, -best_beta, 0.0, 0.0), facts_B_final, ent_B_final, rel_B_final)
    acc_multitask_transferred = eval_retention(best_multi, facts_B_final, ent_B_final, rel_B_final)
    acc_direct_upper_bound = eval_retention(best_B, facts_B_final, ent_B_final, rel_B_final)

    print("=" * 70)
    print(f"ФИНАЛЬНОЕ СРАВНЕНИЕ на held-out задаче B (scale={HELDOUT_SCALE}, N={N_FACTS_FINAL}, БЕЗ переподбора):")
    print(f"  (1) Эволюция на ОДНОЙ задаче A, перенос:      {acc_single_transferred*100:5.1f}%")
    print(f"  (2) Ручная beta на A, перенос:                {acc_handtuned_transferred*100:5.1f}%")
    print(f"  (3) Эволюция на РАСПРЕДЕЛЕНИИ 3 задач, перенос: {acc_multitask_transferred*100:5.1f}%")
    print(f"  (4) Эволюция напрямую на B (upper bound):     {acc_direct_upper_bound*100:5.1f}%")
    print("=" * 70)

    gap_single = acc_direct_upper_bound - acc_single_transferred
    gap_handtuned = acc_direct_upper_bound - acc_handtuned_transferred
    gap_multitask = acc_direct_upper_bound - acc_multitask_transferred
    print(f"Разрыв до upper bound: single-task-эволюция={gap_single*100:.1f} п.п., "
          f"ручная={gap_handtuned*100:.1f} п.п., multi-task-эволюция={gap_multitask*100:.1f} п.п.")

    if gap_multitask < gap_handtuned and gap_multitask < gap_single:
        print("=> Multi-task эволюция переносится ЛУЧШЕ и ручной beta, и single-task эволюции -")
        print("   гипотеза M2 (в multi-task форме из литературы) ПОДТВЕРЖДЕНА, не 'решено' - N мал (400), один held-out.")
    elif gap_multitask < gap_handtuned:
        print("=> Multi-task эволюция лучше ручной beta, но не лучше/сопоставима с single-task эволюцией -")
        print("   частичное подтверждение, не полное.")
    else:
        print("=> Multi-task эволюция НЕ лучше ручной beta при переносе -")
        print("   гипотеза не подтвердилась даже в multi-task форме; M2 остаётся 'приостановлено', теперь с двумя")
        print("   честно проверенными и не сработавшими протоколами (single-task и multi-task).")

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["(1) Single-task\nэволюция A -> B", "(2) Ручная beta\nA -> B",
              "(3) Multi-task\nэволюция -> B", "(4) Прямая эволюция\nна B (upper bound)"]
    vals = [acc_single_transferred*100, acc_handtuned_transferred*100,
            acc_multitask_transferred*100, acc_direct_upper_bound*100]
    colors = ["darkorange", "steelblue", "seagreen", "gray"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel(f"Retention accuracy на held-out задаче B (scale={HELDOUT_SCALE}, %)")
    ax.set_title(f"Single-task vs multi-task эволюция правила пластичности (N={N_FACTS_FINAL})")
    ax.grid(True, axis="y")
    plt.tight_layout()
    path = os.path.join(plots_dir, "evolved_hebbian_multitask_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")

    return acc_single_transferred, acc_handtuned_transferred, acc_multitask_transferred, acc_direct_upper_bound


if __name__ == "__main__":
    run()

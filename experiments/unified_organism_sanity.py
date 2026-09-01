"""
Первый тест ДЕЙСТВИТЕЛЬНО слитой системы: растёт ли ткань, учится ли её
геном (ошибка предсказания должна падать - раньше геном был случайным
и ничему не учился), работает ли быстрая память с приоритетом по метке,
и переживает ли и структура, и память повреждение - вместе, не по
отдельности, как во всех прошлых тестах.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.unified_organism import LivingTissue

torch.manual_seed(42)


def signal(t):
    s = torch.zeros(1, 2, 24, 24)
    s[0, 0, 6:18, 6:18] = 0.5
    return s


def run():
    organism = LivingTissue(size=24, state_dim=16, seed=1)

    print("=" * 70)
    print("1) Рост + обучение генома одновременно (реальная ошибка предсказания):")
    counts, errors = [], []
    for t in range(300):
        n, err = organism.step(sensory_signal=signal(t), train_genome=True)
        counts.append(n)
        errors.append(err)
        if t % 25 == 0 or t == 299:
            print(f"   шаг={t:3d}  живых клеток={n:4d}  средняя ошибка предсказания={err:.4f}")

    print(f"\n   Клеток: {counts[0]} -> {counts[-1]}")
    print(f"   Ошибка предсказания: {errors[0]:.4f} -> {errors[-1]:.4f} "
          f"({'ГЕНОМ УЧИТСЯ' if errors[-1] < errors[10] * 0.7 else 'не сходится'})")

    print("\n2) Быстрая память с приоритетом по метке (synaptic tagging):")
    g = torch.Generator().manual_seed(7)
    fast_dim = organism.fast_dim
    n_facts = 40
    keys = torch.nn.functional.normalize(torch.randn(n_facts, fast_dim, generator=g), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n_facts, fast_dim, generator=g), dim=-1)
    # Половина фактов - "высокая новизна" (сильная метка), половина - "рутина" (слабая метка)
    tags = torch.tensor([1.5 if i % 2 == 0 else 0.3 for i in range(n_facts)])
    for i in range(n_facts):
        organism.write_fact(keys[i], values[i], tag_strength=tags[i].item())

    def decode(vec):
        sims = torch.nn.functional.cosine_similarity(values, vec.unsqueeze(0), dim=-1)
        return int(sims.argmax().item())

    high_tag_correct = low_tag_correct = 0
    for i in range(n_facts):
        pred = organism.read_fact(keys[i])
        correct = (decode(pred) == i)
        if tags[i] > 1.0:
            high_tag_correct += correct
        else:
            low_tag_correct += correct
    print(f"   Высокая метка (новизна): {high_tag_correct}/{n_facts//2} правильно")
    print(f"   Низкая метка (рутина):   {low_tag_correct}/{n_facts//2} правильно")

    print("\n3) СЛИТЫЙ тест на повреждение: структура И память вместе:")
    pre_ablation_count = counts[-1]
    killed = organism.ablate(fraction=0.4)
    alive_after_ablation = int((organism.state[0, 0] > 0.1).sum().item())
    print(f"   Убито клеток: {killed} (было {pre_ablation_count}, осталось {alive_after_ablation})")

    # Память НЕ пострадала физически - W_fast отдельная структура (как
    # гиппокамп отдельная область мозга, не разрушается локальным
    # повреждением коры) - проверяем это явно, не предполагаем.
    post_ablation_correct = sum(decode(organism.read_fact(keys[i])) == i for i in range(n_facts))
    print(f"   Recall фактов СРАЗУ ПОСЛЕ повреждения структуры: {post_ablation_correct}/{n_facts} "
          f"({'память не затронута структурным повреждением' if post_ablation_correct == high_tag_correct + low_tag_correct else 'ЕСТЬ ВЛИЯНИЕ'})")

    recovery_counts = []
    for t in range(150):
        n, err = organism.step(sensory_signal=signal(t), train_genome=True)
        recovery_counts.append(n)
    final_count = recovery_counts[-1]
    recovery_pct = 100.0 * final_count / pre_ablation_count
    print(f"   Восстановление структуры: {alive_after_ablation} -> {final_count} клеток "
          f"({recovery_pct:.1f}% от до-абляционного)")

    final_recall = sum(decode(organism.read_fact(keys[i])) == i for i in range(n_facts))
    print(f"   Recall фактов ПОСЛЕ восстановления структуры: {final_recall}/{n_facts}")
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(errors, color="crimson", linewidth=2)
    axes[0].set_title("Ошибка предсказания генома во время роста (должна падать)")
    axes[0].set_xlabel("Шаг")
    axes[0].set_ylabel("Средняя ошибка предсказания")
    axes[0].grid(True)

    full_counts = counts + [alive_after_ablation] + recovery_counts
    axes[1].plot(full_counts, color="green", linewidth=2)
    axes[1].axvline(len(counts), color="black", linestyle=":", label="Абляция 40%")
    axes[1].set_title("Структурное восстановление слитого организма")
    axes[1].set_xlabel("Шаг")
    axes[1].set_ylabel("Живых клеток")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    path = os.path.join(plots_dir, "unified_organism_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")


if __name__ == "__main__":
    run()

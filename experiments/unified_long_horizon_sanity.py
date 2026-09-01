"""
M6 (ROADMAP.md) - долгая непрерывная работа. НИ РАЗУ не тестировалась за
всю сессию (все прошлые тесты organism - 150-3000 шагов). Здесь - на порядок
дольше (6000 шагов), с непрерывным потоком: рост под меняющимся сигналом,
периодические случайные повреждения (имитация продолжающегося износа среды,
не один разовый ablation-тест), непрерывная запись/чтение фактов в быструю
память между повреждениями. Вопрос: остаётся ли система СТАБИЛЬНОЙ на этом
горизонте (популяция не коллапсирует и не взрывается, ошибка генома не
расходится, память не деградирует катастрофически), или всплывают проблемы,
которые никогда не были видны на коротких тестах?
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.unified_organism import LivingTissue

torch.manual_seed(42)

SIZE = 48
TOTAL_STEPS = 6000
ABLATE_EVERY = 400
FACT_WRITE_EVERY = 50


def signal(t, size):
    s = torch.zeros(1, 2, size, size)
    c = size // 2
    # Дрейфующий по времени центр стимула - не статичный, имитирует
    # меняющуюся среду, а не одну и ту же фиксированную точку вечно.
    r = size // 4
    ox = int(3 * torch.sin(torch.tensor(t * 0.003)).item())
    oy = int(3 * torch.cos(torch.tensor(t * 0.002)).item())
    y0, y1 = max(0, c + oy - r), min(size, c + oy + r)
    x0, x1 = max(0, c + ox - r), min(size, c + ox + r)
    s[0, 0, y0:y1, x0:x1] = 0.5
    return s


def run():
    organism = LivingTissue(size=SIZE, state_dim=16, seed=1)
    g = torch.Generator().manual_seed(7)
    fast_dim = organism.fast_dim
    facts_written = []  # (key, value) - проверяем recall в конце

    counts, errors, ablation_events = [], [], []
    print("=" * 70)
    print(f"Долгий горизонт: {TOTAL_STEPS} шагов, абляция каждые {ABLATE_EVERY}, "
          f"запись фактов каждые {FACT_WRITE_EVERY}")
    for t in range(TOTAL_STEPS):
        n, err = organism.step(sensory_signal=signal(t, SIZE), train_genome=True)
        counts.append(n)
        errors.append(err)

        if t > 0 and t % FACT_WRITE_EVERY == 0:
            key = torch.nn.functional.normalize(torch.randn(fast_dim, generator=g), dim=0)
            value = torch.nn.functional.normalize(torch.randn(fast_dim, generator=g), dim=0)
            organism.write_fact(key, value, tag_strength=1.0)
            facts_written.append((key, value))

        if t > 0 and t % ABLATE_EVERY == 0:
            killed = organism.ablate(fraction=0.3)
            ablation_events.append((t, killed, n))

        if (t + 1) % 1000 == 0:
            recent_err = sum(errors[-200:]) / 200
            print(f"  t={t+1:5d}  население={n:4d}  ошибка(200-avg)={recent_err:.4f}  "
                  f"фактов записано={len(facts_written)}")

    print("\n" + "=" * 70)
    print(f"Циклов абляции: {len(ablation_events)}")
    for t, killed, pre in ablation_events[:5]:
        print(f"   t={t}: было={pre} убито={killed}")
    if len(ablation_events) > 5:
        print(f"   ... и ещё {len(ablation_events)-5} циклов")

    values = torch.stack([v for _, v in facts_written])
    def decode(vec):
        sims = torch.nn.functional.cosine_similarity(values, vec.unsqueeze(0), dim=-1)
        return int(sims.argmax().item())
    correct = sum(decode(organism.read_fact(k)) == i for i, (k, v) in enumerate(facts_written))
    recall_acc = correct / len(facts_written)
    print(f"\nRecall ВСЕХ {len(facts_written)} фактов, записанных за {TOTAL_STEPS} шагов "
          f"(включая записанные в самом начале, до множества повреждений): {correct}/{len(facts_written)} "
          f"({recall_acc*100:.1f}%)")

    print("\n" + "=" * 70)
    print("Честная оценка стабильности:")
    early_pop = sum(counts[:500]) / 500
    late_pop = sum(counts[-500:]) / 500
    early_err = sum(errors[:500]) / 500
    late_err = sum(errors[-500:]) / 500
    print(f"   Население: начало(avg 500)={early_pop:.1f} -> конец(avg 500)={late_pop:.1f}")
    print(f"   Ошибка генома: начало(avg 500)={early_err:.4f} -> конец(avg 500)={late_err:.4f}")
    max_pop = max(counts)
    collapsed = late_pop < 1
    exploded = max_pop > SIZE * SIZE * 0.9
    diverged = late_err > early_err * 3
    print(f"   Коллапс населения: {'ДА' if collapsed else 'нет'}")
    print(f"   Взрыв населения (>90% холста): {'ДА' if exploded else 'нет'} (max={max_pop})")
    print(f"   Расхождение ошибки генома (>3x): {'ДА' if diverged else 'нет'}")
    print(f"   Recall памяти после долгой работы+повреждений: {recall_acc*100:.1f}%")
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(counts, linewidth=0.7, color="green")
    for t, _, _ in ablation_events:
        axes[0].axvline(t, color="red", alpha=0.15, linewidth=0.5)
    axes[0].set_title(f"Население за {TOTAL_STEPS} шагов ({len(ablation_events)} циклов абляции)")
    axes[0].set_xlabel("Шаг")
    axes[0].grid(True)
    axes[1].plot(errors, linewidth=0.3, color="crimson", alpha=0.6)
    axes[1].set_title("Ошибка предсказания генома")
    axes[1].set_xlabel("Шаг")
    axes[1].grid(True)
    plt.tight_layout()
    path = os.path.join(plots_dir, "unified_long_horizon_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")

    return counts, errors, recall_acc


if __name__ == "__main__":
    run()

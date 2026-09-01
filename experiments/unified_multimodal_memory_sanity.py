"""
Мердж columnar_voting.py (Thousand-Brains-стиль голосование колонок) В
core/unified_organism.py, приоритет (a) из README/ROADMAP "Текущее состояние" -
до сих пор LivingTissue.write_fact/read_fact были ОДНОЙ общей плотной SDR-
колонкой на весь организм, columnar_voting.py тестировал голосование отдельно,
как самостоятельный скрипт, никогда не был частью самого организма.

Что тестируем ЧЕСТНО, не просто "код скомпилировался":
организм теперь умеет писать ОДИН И ТОТ ЖЕ факт через ДВЕ независимые
модальности (например: ключ, выведенный из пространственного контекста ткани,
и отдельный компактный "символьный" ключ) - и читать факт слиянием голосов
обеих колонок БЕЗ ручного веса на модальность (read_fact_voted), как и
в columnar_voting.py, но теперь для непрерывного значения, а не дискретного
кандидата (weight = уверенность/норма предсказания колонки, не константа).

Протокол:
1. N фактов пишутся под ОБЕИМИ модальностями (spatial-key, symbolic-key) ->
   одно и то же value.
2. Чистый recall (без шума) - sanity, обе колонки и voting должны работать.
3. ГЛАВНЫЙ тест: одна модальность (spatial) получает гауссов шум В КЛЮЧЕ на
   чтении (имитация деградировавшего/шумного сенсорного канала - ровно то,
   что требует пункт "протестировать устойчивость к шумному потоку" из
   README). Сравниваем:
     - spatial-only (должен деградировать - baseline "без слияния вообще")
     - symbolic-only (не должен деградировать - контроль, что шум ЛОКАЛЕН)
     - naive-average (0.5/0.5 без учёта уверенности) - более сильный baseline,
       чем "без слияния", но всё ещё ручной вес
     - read_fact_voted (confidence-weighted, наша реализация)
   Гипотеза: voted >= naive-average >= spatial-only при зашумлении одной
   модальности, потому что шумный ключ обычно даёт слабый (низкая норма)
   отклик колонки -> voted естественно понижает её вклад, тогда как
   naive-average слепо держит вес 0.5 у деградировавшей колонки.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.unified_organism import LivingTissue

torch.manual_seed(42)


def run():
    organism = LivingTissue(size=8, state_dim=16, seed=1)
    fast_dim = organism.fast_dim  # 14

    SPATIAL_KEY_DIM = fast_dim   # ключ, "выведенный из контекста ткани"
    SYMBOLIC_KEY_DIM = 8         # компактный внешний символьный ключ
    organism.add_modality("spatial", key_dim=SPATIAL_KEY_DIM, seed=11)
    organism.add_modality("symbolic", key_dim=SYMBOLIC_KEY_DIM, seed=22)

    g = torch.Generator().manual_seed(7)
    N = 40
    spatial_keys = torch.nn.functional.normalize(torch.randn(N, SPATIAL_KEY_DIM, generator=g), dim=-1)
    symbolic_keys = torch.nn.functional.normalize(torch.randn(N, SYMBOLIC_KEY_DIM, generator=g), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(N, fast_dim, generator=g), dim=-1)

    for i in range(N):
        organism.write_fact_modal("spatial", spatial_keys[i], values[i], tag_strength=1.0)
        organism.write_fact_modal("symbolic", symbolic_keys[i], values[i], tag_strength=1.0)

    def decode(vec):
        sims = torch.nn.functional.cosine_similarity(values, vec.unsqueeze(0), dim=-1)
        return int(sims.argmax().item())

    def acc(pred_fn):
        return sum(decode(pred_fn(i)) == i for i in range(N)) / N

    print("=" * 70)
    print(f"1) ЧИСТЫЙ recall (N={N} фактов, без шума):")
    spatial_clean = acc(lambda i: organism.read_fact_modal("spatial", spatial_keys[i]))
    symbolic_clean = acc(lambda i: organism.read_fact_modal("symbolic", symbolic_keys[i]))
    voted_clean = acc(lambda i: organism.read_fact_voted(
        {"spatial": spatial_keys[i], "symbolic": symbolic_keys[i]}))
    print(f"   spatial-only:  {spatial_clean*100:5.1f}%")
    print(f"   symbolic-only: {symbolic_clean*100:5.1f}%")
    print(f"   voted (both):  {voted_clean*100:5.1f}%")

    print("\n2) ШУМНАЯ spatial-модальность (гауссов шум в ключе на чтении, sigma перебираем):")
    print("   (устойчивость к деградировавшему сенсорному каналу - README пункт 'шумный поток')")
    for sigma in [0.3, 0.6, 1.0, 1.5]:
        gn = torch.Generator().manual_seed(99)
        noisy_spatial_keys = torch.nn.functional.normalize(
            spatial_keys + sigma * torch.randn(N, SPATIAL_KEY_DIM, generator=gn), dim=-1)

        spatial_noisy = acc(lambda i: organism.read_fact_modal("spatial", noisy_spatial_keys[i]))
        symbolic_ctrl = acc(lambda i: organism.read_fact_modal("symbolic", symbolic_keys[i]))
        naive_avg = acc(lambda i: 0.5 * organism.read_fact_modal("spatial", noisy_spatial_keys[i])
                         + 0.5 * organism.read_fact_modal("symbolic", symbolic_keys[i]))
        voted = acc(lambda i: organism.read_fact_voted(
            {"spatial": noisy_spatial_keys[i], "symbolic": symbolic_keys[i]}))
        print(f"   sigma={sigma:.1f}: spatial-only={spatial_noisy*100:5.1f}%  "
              f"symbolic-only(ctrl)={symbolic_ctrl*100:5.1f}%  "
              f"naive-avg={naive_avg*100:5.1f}%  voted={voted*100:5.1f}%")

    print("=" * 70)
    print("Честный вывод: смотрим, ЛЕЖИТ ли voted >= naive-avg >= spatial-only при всех sigma,")
    print("и приближается ли voted к symbolic-only (контролю) по мере роста sigma -")
    print("если нет, гипотеза о confidence-weighting как реальном преимуществе не подтверждается")
    print("=" * 70)


if __name__ == "__main__":
    run()

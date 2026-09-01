"""
M8(b) (docs/ROADMAP.md): "VSA/колончатая память на систематическом масштабе" -
колончатая часть. Продолжение unified_multimodal_memory_sanity.py (тот
прогон: 2 модальности, ОДИН seed, шум добавлялся в ОДНУ модальность за раз)
на систематическом масштабе - 2..8 ОДНОВРЕМЕННЫХ модальностей, N=5 seed.

Отличие постановки: вместо "1 шумный канал + 1 чистый контроль" здесь у
КАЖДОЙ из 8 модальностей своя ФИКСИРОВАННАЯ интенсивность шума (растущая
по индексу: модальность 0 чистая, модальность 7 самая шумная) - имитация
реалистичного набора сенсоров разного качества, ОДНОВРЕМЕННО активных, а
не один нарочно выделенный "плохой" канал. Метрика ROADMAP - деградация
recall на КАЖДУЮ добавленную модальность: сравниваем voted (confidence-
weighted) с naive-average и best-single КАК ФУНКЦИЮ m (число одновременно
голосующих модальностей), а не при фиксированном m=2.

Гипотеза: voted >= naive-avg >= best-single ПРИ ЛЮБОМ m от 2 до 8 (не
только при m=2, как проверено раньше), и разрыв voted-naive_avg не
схлопывается к нулю по мере роста m (больше шумных каналов должно означать
БОЛЬШЕ пользы от confidence-weighting, не меньше - naive-avg разбавляется
шумом линейно, voted должен естественно приглушать плохие каналы).
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics as st
import torch
import torch.nn.functional as F
from core.unified_organism import LivingTissue

N_FACTS = 60
N_MODALITIES = 8
KEY_DIMS = [14, 8, 10, 6, 12, 9, 11, 7]
SIGMAS = [0.25 * i for i in range(N_MODALITIES)]  # модальность 0 чистая, 7 - самая шумная
SEEDS = [1, 2, 3, 4, 5]


def run_seed(seed):
    organism = LivingTissue(size=8, state_dim=16, seed=seed)
    fast_dim = organism.fast_dim
    names = [f"mod{i}" for i in range(N_MODALITIES)]
    for i, name in enumerate(names):
        organism.add_modality(name, key_dim=KEY_DIMS[i], seed=seed * 100 + i)

    g = torch.Generator().manual_seed(seed * 7 + 1)
    clean_keys = [F.normalize(torch.randn(N_FACTS, KEY_DIMS[i], generator=g), dim=-1)
                  for i in range(N_MODALITIES)]
    values = F.normalize(torch.randn(N_FACTS, fast_dim, generator=g), dim=-1)

    for i, name in enumerate(names):
        for f_idx in range(N_FACTS):
            organism.write_fact_modal(name, clean_keys[i][f_idx], values[f_idx], tag_strength=1.0)

    gn = torch.Generator().manual_seed(seed * 13 + 5)
    noisy_keys = [F.normalize(clean_keys[i] + SIGMAS[i] * torch.randn(N_FACTS, KEY_DIMS[i], generator=gn), dim=-1)
                  for i in range(N_MODALITIES)]

    def decode(vec):
        sims = F.cosine_similarity(values, vec.unsqueeze(0), dim=-1)
        return int(sims.argmax().item())

    def acc(pred_fn):
        return sum(decode(pred_fn(idx)) == idx for idx in range(N_FACTS)) / N_FACTS

    per_m = {}
    for m in range(2, N_MODALITIES + 1):
        used = names[:m]
        singles = [acc(lambda idx, i=i: organism.read_fact_modal(names[i], noisy_keys[i][idx]))
                   for i in range(m)]
        best_single = max(singles)
        naive_avg = acc(lambda idx: sum(organism.read_fact_modal(names[i], noisy_keys[i][idx])
                                         for i in range(m)) / m)
        voted = acc(lambda idx: organism.read_fact_voted(
            {names[i]: noisy_keys[i][idx] for i in range(m)}))
        per_m[m] = (best_single, naive_avg, voted)
    return per_m


if __name__ == "__main__":
    print("=" * 70)
    print(f"M8(b): голосование колонок, m=2..{N_MODALITIES} одновременных модальностей, "
          f"N={len(SEEDS)} seed, N_FACTS={N_FACTS}")
    print(f"sigma по модальности (0=чистая): {[round(s,2) for s in SIGMAS]}")
    print("=" * 70)

    all_runs = [run_seed(s) for s in SEEDS]
    for m in range(2, N_MODALITIES + 1):
        bs = [r[m][0] for r in all_runs]
        na = [r[m][1] for r in all_runs]
        vo = [r[m][2] for r in all_runs]
        mbs, mna, mvo = st.mean(bs), st.mean(na), st.mean(vo)
        gap_vs_naive = (mvo - mna) * 100
        gap_vs_best = (mvo - mbs) * 100
        print(f"m={m}: best-single={mbs*100:5.1f}%  naive-avg={mna*100:5.1f}%  "
              f"voted={mvo*100:5.1f}%   gap(voted-naive)={gap_vs_naive:+5.1f}п.п.  "
              f"gap(voted-best)={gap_vs_best:+5.1f}п.п.")
    print("=" * 70)

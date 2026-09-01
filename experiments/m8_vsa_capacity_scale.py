"""
M8(a) (docs/ROADMAP.md): "VSA/колончатая память на систематическом масштабе" -
VSA-часть. Продолжение unified_vsa_compositional_sanity.py (тот прогон:
N_ENTITIES=150, K до 32, ОДИН seed) на масштабе на порядок больше по ДВУМ
независимым осям, N>=5 seed на точку (там сида не было вообще):

  (A) N_ENTITIES: 150 -> 2000 (K=4 фиксирован - там, где разрыв был самым
      чистым и не искажён потолком ёмкости, см. VERIFICATION_LOG).
  (B) K (пар/объект): 32 -> 256 (N_ENTITIES=150 фиксирован, как в
      исходном тесте).

Отличие от исходного теста: там N_ROLES=8 было ФИКСИРОВАНО, поэтому при
K>8 роли переиспользовались внутри одного bundle (сталкивались друг с
другом) - смешивало "ёмкость VSA-связывания" с "нехватку ролей". Здесь
N_ROLES=K на каждой точке K-развёртки (роли всегда различны в пределах
одного объекта) - изолирует именно ёмкость, не путает её с искусственной
коллизией ролей. Явно отмечено, т.к. меняет сравнимость с прошлым тестом.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics as st
import torch
from core.unified_organism import LivingTissue
from core.vsa_binding import random_vectors, circular_conv

VSA_DIM = 256
FILLER_VOCAB = 40
SEEDS = [1, 2, 3, 4, 5]


def run_point(n_entities, K, seed):
    n_roles = K
    g = torch.Generator().manual_seed(seed)
    entity_keys = random_vectors(n_entities, VSA_DIM, seed=seed)
    filler_vocab = random_vectors(FILLER_VOCAB, VSA_DIM, seed=seed + 1)

    def decode(vec):
        sims = torch.nn.functional.cosine_similarity(filler_vocab, vec.unsqueeze(0), dim=-1)
        return int(sims.argmax().item())

    gk = torch.Generator().manual_seed(seed * 1000 + K)
    filler_idx = torch.randint(0, FILLER_VOCAB, (n_entities, K), generator=gk)

    organism_c = LivingTissue(size=4, state_dim=16, seed=seed)
    organism_c.init_vsa(vsa_dim=VSA_DIM, n_roles=n_roles, seed=seed)
    for e in range(n_entities):
        pairs = [(r, filler_vocab[filler_idx[e, r].item()]) for r in range(K)]
        organism_c.write_compositional(entity_keys[e], pairs, tag_strength=1.0)
    correct_c = total_c = 0
    for e in range(n_entities):
        for r in range(K):
            pred = organism_c.read_compositional(entity_keys[e], r)
            correct_c += int(decode(pred) == filler_idx[e, r].item())
            total_c += 1
    acc_c = correct_c / total_c

    organism_f = LivingTissue(size=4, state_dim=16, seed=seed)
    organism_f.add_modality("flat_compositional", key_dim=VSA_DIM, sdr_dim=512, seed=seed, value_dim=VSA_DIM)
    role_vecs = organism_c.vsa_roles
    for e in range(n_entities):
        for r in range(K):
            flat_key = circular_conv(entity_keys[e], role_vecs[r])
            organism_f.write_fact_modal("flat_compositional", flat_key,
                                         filler_vocab[filler_idx[e, r].item()], tag_strength=1.0)
    correct_f = total_f = 0
    for e in range(n_entities):
        for r in range(K):
            flat_key = circular_conv(entity_keys[e], role_vecs[r])
            pred = organism_f.read_fact_modal("flat_compositional", flat_key)
            correct_f += int(decode(pred) == filler_idx[e, r].item())
            total_f += 1
    acc_f = correct_f / total_f
    return acc_c, acc_f


def sweep(label, points, fixed_desc):
    print(f"\n=== {label} ({fixed_desc}, N={len(SEEDS)} seed) ===")
    for n_entities, K in points:
        cs, fs = [], []
        for seed in SEEDS:
            c, f = run_point(n_entities, K, seed)
            cs.append(c)
            fs.append(f)
        mc, mf = st.mean(cs), st.mean(fs)
        sc = st.stdev(cs) if len(cs) > 1 else 0.0
        sf = st.stdev(fs) if len(fs) > 1 else 0.0
        gap = (mc - mf) * 100
        print(f"n_entities={n_entities:5d} K={K:4d}: "
              f"compositional={mc*100:5.1f}%(sd{sc*100:4.1f}) "
              f"flat={mf*100:5.1f}%(sd{sf*100:4.1f})  gap={gap:+6.1f}п.п.")


if __name__ == "__main__":
    print("=" * 70)
    print("M8(a): VSA compositional vs flat, систематический масштаб, N=5 seed/точку")
    sweep("(A) Ось N_ENTITIES (K=4 фикс.)",
          [(150, 4), (500, 4), (1000, 4), (2000, 4)],
          "K=4")
    sweep("(B) Ось K (N_ENTITIES=150 фикс., N_ROLES=K - без коллизий)",
          [(150, 32), (150, 64), (150, 128), (150, 256)],
          "N_ENTITIES=150")
    print("=" * 70)

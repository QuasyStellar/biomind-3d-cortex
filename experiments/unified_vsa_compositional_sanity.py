"""
Мердж VSA-binding (vsa_binding.py) в core/unified_organism.py (приоритет (a)) -
последний оставшийся пункт из "колонки, JEPA-понимание, phase-binding".
VSA-ёмкость (K до 512 на dim=1024) и SDR-память с synaptic tagging уже
проверены каждая ПО ОТДЕЛЬНОСТИ - здесь они впервые СОВМЕЩЕНЫ в самом
организме: `LivingTissue.write_compositional()`/`read_compositional()`.

Вопрос, который не задавался раньше: если у "объекта" несколько атрибутов
(role-filler пар), эффективнее ли связать их через circular convolution в
ОДИН bundle и сделать ОДНУ SDR-запись (compositional), чем писать каждую
пару ОТДЕЛЬНОЙ SDR-записью (flat, через уже смерженный write_fact_modal)?
Гипотеза: compositional тратит SDR-бюджет (число записей в W) экономнее -
1 запись на объект вместо K, что должно снижать интерференцию SDR-памяти
при большом числе объектов, ЦЕНОЙ шума самого VSA-развязывания.

Протокол: N_ENTITIES объектов, каждый с K role-filler парами (roles
переиспользуются across объектов - разные "слоты" одного и того же
объекта, roles ФИКСИРОВАНЫ, fillers - из словаря FILLER_VOCAB). Сравниваем
recall КАЖДОЙ отдельной (entity, role) пары:
  - compositional: 1 SDR-запись на entity (bundle из K пар)
  - flat: K SDR-записей на entity (по одной на пару, ключ = circular_conv(entity_key, role))
на РАВНОМ количестве объектов, при РАЗНОМ числе SDR-записей (K:1 бюджет).
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.unified_organism import LivingTissue
from core.vsa_binding import random_vectors, circular_conv

torch.manual_seed(42)

VSA_DIM = 256
N_ROLES = 8
N_ENTITIES = 60
FILLER_VOCAB = 40


def run(seed=1):
    g = torch.Generator().manual_seed(seed)
    entity_keys = random_vectors(N_ENTITIES, VSA_DIM, seed=seed)
    filler_vocab = random_vectors(FILLER_VOCAB, VSA_DIM, seed=seed + 1)

    def decode(vec):
        sims = torch.nn.functional.cosine_similarity(filler_vocab, vec.unsqueeze(0), dim=-1)
        return int(sims.argmax().item())

    results = {"compositional": [], "flat": []}
    for K in [1, 2, 3, 4, 6]:
        gk = torch.Generator().manual_seed(seed * 1000 + K)
        filler_idx = torch.randint(0, FILLER_VOCAB, (N_ENTITIES, K), generator=gk)

        # --- compositional: 1 SDR-запись на entity ---
        organism_c = LivingTissue(size=4, state_dim=16, seed=seed)
        organism_c.init_vsa(vsa_dim=VSA_DIM, n_roles=N_ROLES, seed=seed)
        for e in range(N_ENTITIES):
            pairs = [(r, filler_vocab[filler_idx[e, r].item()]) for r in range(K)]
            organism_c.write_compositional(entity_keys[e], pairs, tag_strength=1.0)
        correct_c = total_c = 0
        for e in range(N_ENTITIES):
            for r in range(K):
                pred = organism_c.read_compositional(entity_keys[e], r)
                correct_c += int(decode(pred) == filler_idx[e, r].item())
                total_c += 1
        acc_c = correct_c / total_c

        # --- flat: K отдельных SDR-записей на entity (ключ = entity ⊛ role) ---
        organism_f = LivingTissue(size=4, state_dim=16, seed=seed)
        organism_f.add_modality("flat_compositional", key_dim=VSA_DIM, sdr_dim=512, seed=seed, value_dim=VSA_DIM)
        role_vecs = organism_c.vsa_roles  # переиспользуем те же role-векторы для честности
        for e in range(N_ENTITIES):
            for r in range(K):
                flat_key = circular_conv(entity_keys[e], role_vecs[r])
                organism_f.write_fact_modal("flat_compositional", flat_key,
                                             filler_vocab[filler_idx[e, r].item()], tag_strength=1.0)
        correct_f = total_f = 0
        for e in range(N_ENTITIES):
            for r in range(K):
                flat_key = circular_conv(entity_keys[e], role_vecs[r])
                pred = organism_f.read_fact_modal("flat_compositional", flat_key)
                correct_f += int(decode(pred) == filler_idx[e, r].item())
                total_f += 1
        acc_f = correct_f / total_f

        n_writes_c, n_writes_f = N_ENTITIES, N_ENTITIES * K
        print(f"K={K}: compositional={acc_c*100:5.1f}% ({n_writes_c} SDR-записей)   "
              f"flat={acc_f*100:5.1f}% ({n_writes_f} SDR-записей)")
        results["compositional"].append(acc_c)
        results["flat"].append(acc_f)

    return results


if __name__ == "__main__":
    print("=" * 70)
    print(f"N_ENTITIES={N_ENTITIES}, VSA_DIM={VSA_DIM}, FILLER_VOCAB={FILLER_VOCAB}")
    r = run()
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    Ks = [1, 2, 3, 4, 6]
    ax.plot(Ks, [v * 100 for v in r["compositional"]], "o-", label="compositional (1 запись/entity)", color="darkorange")
    ax.plot(Ks, [v * 100 for v in r["flat"]], "o-", label="flat (K записей/entity)", color="steelblue")
    ax.set_xlabel("K (role-filler пар на объект)")
    ax.set_ylabel("Recall accuracy (%)")
    ax.set_title("VSA-связывание + SDR-запись: compositional vs flat (мердж в organism)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(plots_dir, "unified_vsa_compositional_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")

"""
Проверка гипотезы M3 (ROADMAP.md): голосующие идентичные колонки должны
НЕ деградировать (в идеале — улучшаться) при росте числа одновременных
модальностей, в отличие от ручной формулы слияния, откалиброванной под
фиксированное число входов.

Мир: N_CONCEPTS общих понятий. Каждая модальность - зашумлённая проекция
понятия через СВОЙ фиксированный "сенсорный" линейный преобразователь
(разные ракурсы одного и того же объекта). Одна модальность в одиночку
не идеальна (шум) - вопрос в том, помогает ли объединение нескольких.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.columnar_voting import Column, vote_consensus

torch.manual_seed(42)

DIM = 64
N_CONCEPTS = 150
N_MODALITIES = 4
TRAIN_NOISE = 0.15
# 0.45 оказался слишком суровым - загонял всё в шумовой пол (макс 27.5% даже
# при K=4), не давая увидеть форму кривой. 0.25 даёт чистый, интерпретируемый
# диапазон (K=1: 25%, K=4: 83.5% на калибровочном прогоне).
TEST_NOISE = 0.25


def build_world(seed=42):
    g = torch.Generator().manual_seed(seed)
    object_vecs = torch.nn.functional.normalize(torch.randn(N_CONCEPTS, DIM, generator=g), dim=-1)
    modality_transforms = [torch.linalg.qr(torch.randn(DIM, DIM, generator=g))[0] for _ in range(N_MODALITIES)]
    return object_vecs, modality_transforms


def cue(concept_idx, modality, object_vecs, modality_transforms, noise, g):
    raw = object_vecs[concept_idx] @ modality_transforms[modality]
    raw = raw + noise * torch.randn(DIM, generator=g)
    return torch.nn.functional.normalize(raw, dim=0)


def run():
    object_vecs, modality_transforms = build_world()
    g = torch.Generator().manual_seed(1)

    columns = [Column(dim=DIM, sdr_dim=512, sparsity=0.08, beta=0.9, seed=10 + m) for m in range(N_MODALITIES)]

    # 1-shot обучение: одна запись на понятие на модальность
    for c in range(N_CONCEPTS):
        for m in range(N_MODALITIES):
            example = cue(c, m, object_vecs, modality_transforms, TRAIN_NOISE, g)
            columns[m].learn(example, object_vecs[c])

    # --- Тест: голосование, K одновременных модальностей от 1 до 4 ---
    print("=" * 70)
    print("Голосование колонок (равноправное, без ручных весов):")
    voting_results = {}
    for K in range(1, N_MODALITIES + 1):
        correct = 0
        trials = 300
        for _ in range(trials):
            c = torch.randint(0, N_CONCEPTS, (1,), generator=g).item()
            active_modalities = torch.randperm(N_MODALITIES, generator=g)[:K].tolist()
            cues = {m: cue(c, m, object_vecs, modality_transforms, TEST_NOISE, g) for m in active_modalities}
            pred = vote_consensus(columns, cues, object_vecs)
            correct += (pred == c)
        acc = correct / trials
        voting_results[K] = acc
        print(f"  K={K} одновременных модальностей -> accuracy={acc*100:5.1f}%")

    # --- Baseline: ручная формула, откалиброванная под K=2 (веса 0.6/0.4),
    #     игнорирует любые модальности сверх этих двух слотов ---
    print("\nРучная формула слияния (жёстко под 2 модальности, веса 0.6/0.4):")
    fixed_results = {}
    for K in range(1, N_MODALITIES + 1):
        correct = 0
        trials = 300
        for _ in range(trials):
            c = torch.randint(0, N_CONCEPTS, (1,), generator=g).item()
            active_modalities = torch.randperm(N_MODALITIES, generator=g)[:K].tolist()
            # Формула физически имеет только 2 слота - модальности сверх двух
            # структурно не могут повлиять на результат, как и было бы у
            # захардкоженной формулы слияния в v1-стиле кода
            slots = active_modalities[:2]
            preds = []
            weights = [0.6, 0.4]
            combined = torch.zeros(DIM)
            for i, m in enumerate(slots):
                c_vec = cue(c, m, object_vecs, modality_transforms, TEST_NOISE, g)
                pred_vec = columns[m].memory.read(c_vec)
                combined = combined + weights[i] * pred_vec
            sims = torch.nn.functional.cosine_similarity(object_vecs, combined.unsqueeze(0), dim=-1)
            pred = int(sims.argmax().item())
            correct += (pred == c)
        acc = correct / trials
        fixed_results[K] = acc
        print(f"  K={K} одновременных модальностей -> accuracy={acc*100:5.1f}% "
              f"({'использует все' if K<=2 else f'ИГНОРИРУЕТ {K-2} модальности - структурно не может'})")
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(voting_results.keys()), [v * 100 for v in voting_results.values()],
            "g-o", linewidth=2.5, label="Голосование колонок (равноправное)")
    ax.plot(list(fixed_results.keys()), [v * 100 for v in fixed_results.values()],
            "r--s", linewidth=2, label="Ручная формула (жёстко под K=2)")
    ax.set_xlabel("K — число одновременных модальностей")
    ax.set_ylabel("Recall accuracy (%)")
    ax.set_title("Голосование колонок vs ручная формула слияния при росте числа модальностей")
    ax.set_xticks(list(range(1, N_MODALITIES + 1)))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(plots_dir, "columnar_voting_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")

    return voting_results, fixed_results


if __name__ == "__main__":
    run()

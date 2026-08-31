"""
Сон/кристаллизация с нуля, без backprop нигде: день — факты пишутся в
SDR-гиппокамп (уже проверенный компонент 2). Ночь — реплей дневных фактов
+ выборка старых (rehearsal, не "генеративный" реплей — генеративной
модели у нас пока нет, не переоцениваем то, чего не построили) обучает
PC-кору (predictive_coding.py, компонент из первой проверки) через
локальную релаксацию, НЕ backprop+Adam, как было в v1. После ночи —
гиппокамп обнуляется (кристаллизация).

Сравнение: без сна (факты только в гиппокампе, упираются в его ёмкость —
мы уже знаем эту кривую из hippocampus_retention_sweep.py) vs со сном
(перенос в PC-кору, которая не имеет той же жёсткой ёмкостной стены).
"""
import sys, os, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.sdr_hippocampus import SDRHippocampus
from core.predictive_coding import PredictiveCodingNet

torch.manual_seed(42)
random.seed(42)

DIM = 64
N_ENTITIES = 400
N_RELATIONS = 30
DAYS = 10
FACTS_PER_DAY = 100


def make_all_facts(n, seed=42):
    g = torch.Generator().manual_seed(seed)
    entity_vecs = torch.nn.functional.normalize(torch.randn(N_ENTITIES, DIM, generator=g), dim=-1)
    relation_vecs = torch.nn.functional.normalize(torch.randn(N_RELATIONS, DIM, generator=g), dim=-1)
    seen = set()
    facts = []
    while len(facts) < n:
        s = torch.randint(0, N_ENTITIES, (1,), generator=g).item()
        r = torch.randint(0, N_RELATIONS, (1,), generator=g).item()
        o = torch.randint(0, N_ENTITIES, (1,), generator=g).item()
        if s == o or (s, r) in seen:
            continue
        seen.add((s, r))
        facts.append((s, r, o))
    return facts, entity_vecs, relation_vecs


def key_of(s, r, entity_vecs, relation_vecs):
    return torch.nn.functional.normalize(entity_vecs[s] + relation_vecs[r], dim=0)


def decode(vec, entity_vecs):
    sims = entity_vecs @ vec
    return int(sims.argmax().item())


def eval_hippo(hippo, facts, entity_vecs, relation_vecs):
    correct = 0
    for s, r, o in facts:
        pred = hippo.read(key_of(s, r, entity_vecs, relation_vecs))
        if decode(pred, entity_vecs) == o:
            correct += 1
    return correct / len(facts)


def eval_pc(pc, facts, entity_vecs, relation_vecs):
    correct = 0
    for s, r, o in facts:
        key = key_of(s, r, entity_vecs, relation_vecs).unsqueeze(0)
        pred = pc.forward_pass(key).squeeze(0)
        if decode(pred, entity_vecs) == o:
            correct += 1
    return correct / len(facts)


def run():
    all_facts, entity_vecs, relation_vecs = make_all_facts(DAYS * FACTS_PER_DAY)

    # --- Условие A: без сна, факты только в гиппокампе (упирается в его ёмкость) ---
    hippo_no_sleep = SDRHippocampus(dim=DIM, sdr_dim=1024, sparsity=0.06, beta=0.9, seed=1)

    # --- Условие B: со сном - гиппокамп (день) + PC-кора (ночь), zero backward() ---
    hippo_sleep = SDRHippocampus(dim=DIM, sdr_dim=1024, sparsity=0.06, beta=0.9, seed=1)
    # relax_lr=0.1 расходился на этой задаче (энергия росла после ~40 шагов
    # обучения, |W| неограниченно росла) - найдено и исправлено: меньший
    # шаг релаксации + weight_decay дают гладкую, стабильную сходимость.
    cortex = PredictiveCodingNet([DIM, 128, DIM], relax_steps=60, relax_lr=0.05, weight_lr=0.006, seed=2, adam=True, weight_decay=0.03)

    accumulated = []
    replay_archive = []
    no_sleep_curve, sleep_curve = [], []

    for day in range(1, DAYS + 1):
        day_facts = all_facts[(day - 1) * FACTS_PER_DAY: day * FACTS_PER_DAY]
        accumulated.extend(day_facts)

        # WAKE: оба условия пишут факты дня в гиппокамп (zero backward, уже проверено)
        for s, r, o in day_facts:
            key = key_of(s, r, entity_vecs, relation_vecs)
            hippo_no_sleep.write(key, entity_vecs[o])
            hippo_sleep.write(key, entity_vecs[o])

        # SLEEP (только условие B): rehearsal-реплей дня + выборка старых фактов,
        # обучение PC-коры локальной релаксацией, без единого .backward()
        replay_batch = list(day_facts) + random.sample(replay_archive, min(len(replay_archive), 80))
        Xb = torch.stack([key_of(s, r, entity_vecs, relation_vecs) for s, r, o in replay_batch])
        Yb = torch.stack([entity_vecs[o] for s, r, o in replay_batch])
        for _ in range(15):
            cortex.train_step(Xb, Yb)
        replay_archive.extend(day_facts)
        hippo_sleep.W.zero_()  # кристаллизация: буфер обнуляется после переноса

        # AUDIT: recall по ВСЕМ фактам, накопленным с Дня 1
        acc_no_sleep = eval_hippo(hippo_no_sleep, accumulated, entity_vecs, relation_vecs)
        acc_sleep = eval_pc(cortex, accumulated, entity_vecs, relation_vecs)
        no_sleep_curve.append((day, acc_no_sleep))
        sleep_curve.append((day, acc_sleep))
        print(f"День {day:2d}/{DAYS} | Всего фактов: {len(accumulated):4d} | "
              f"Без сна (гиппокамп): {acc_no_sleep*100:5.1f}% | "
              f"Со сном (PC-кора): {acc_sleep*100:5.1f}%")

    print("=" * 70)
    print(f"Финал (N={len(accumulated)}): без сна={no_sleep_curve[-1][1]*100:.1f}%  "
          f"со сном={sleep_curve[-1][1]*100:.1f}%  "
          f"разница={{:+.1f}} п.п.".format((sleep_curve[-1][1]-no_sleep_curve[-1][1])*100))
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(*zip(*no_sleep_curve), "r--s", label=f"Без сна ({no_sleep_curve[-1][1]*100:.1f}%)", linewidth=2)
    ax.plot(*zip(*sleep_curve), "g-o", label=f"Со сном, PC-релаксация, zero backward ({sleep_curve[-1][1]*100:.1f}%)", linewidth=2.5)
    ax.set_xlabel("День (100 новых фактов/день)")
    ax.set_ylabel("Cumulative recall accuracy (%)")
    ax.set_title("Сон/кристаллизация на PC-релаксации (без backprop) vs без сна")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(plots_dir, "sleep_consolidation_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")
    return no_sleep_curve, sleep_curve


if __name__ == "__main__":
    run()

"""
Переработанный тест сна: не "кто лучше зубрит список", а честная проверка
того, для чего сон нужен биологически — обобщение схемы на НИКОГДА не
виденные сущности, а не запоминание записанного.

Мир: категории сущностей с общим правилом (категория -> объект по
отношению), плюс доля исключений (произвольные факты, нарушающие правило).

Гиппокамп физически не может ответить на факт, которого не писал -
это не его слабость, это структурное свойство lookup-памяти. Кора,
если "сон" реально извлекает схему, должна уметь отвечать на такие
факты про новые сущности той же категории (zero-shot generalization).

Два теста, а не один:
  (a) Recall исключений (гиппокамп должен выигрывать - его работа)
  (b) Zero-shot на новых сущностях известной категории (кора должна
      выигрывать, если консолидация действительно извлекла схему)
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
N_CATEGORIES = 8
ENTITIES_PER_CATEGORY = 60
N_RELATIONS = 5
EXCEPTION_RATE = 0.15  # доля фактов, нарушающих правило категории
CATEGORY_NOISE = 0.35  # насколько энтити внутри категории разбросаны вокруг центроида


def build_world(seed=42):
    g = torch.Generator().manual_seed(seed)
    category_centroids = torch.nn.functional.normalize(torch.randn(N_CATEGORIES, DIM, generator=g), dim=-1)
    relation_vecs = torch.nn.functional.normalize(torch.randn(N_RELATIONS, DIM, generator=g), dim=-1)
    # Объекты - тоже отдельный пул (то, во что превращается subject+relation)
    N_OBJECTS = 60
    object_vecs = torch.nn.functional.normalize(torch.randn(N_OBJECTS, DIM, generator=g), dim=-1)

    # Каждая (категория, отношение) пара имеет ОДИН фиксированный "правильный" объект - это и есть схема
    rule_object = torch.randint(0, N_OBJECTS, (N_CATEGORIES, N_RELATIONS), generator=g)

    # Сущности: у каждой категории свой центроид + шум - создаёт кластерную структуру,
    # которую обобщающая сеть МОЖЕТ выучить (в отличие от чисто случайных векторов)
    entity_vecs = []
    entity_category = []
    for c in range(N_CATEGORIES):
        for _ in range(ENTITIES_PER_CATEGORY):
            v = torch.nn.functional.normalize(
                category_centroids[c] + torch.randn(DIM, generator=g) * CATEGORY_NOISE, dim=0)
            entity_vecs.append(v)
            entity_category.append(c)
    entity_vecs = torch.stack(entity_vecs)
    entity_category = torch.tensor(entity_category)

    return entity_vecs, entity_category, relation_vecs, object_vecs, rule_object


def make_fact(subj_idx, rel_idx, entity_category, rule_object, exception=False, g=None):
    c = entity_category[subj_idx].item()
    if exception:
        obj_idx = torch.randint(0, rule_object.max().item() + 1, (1,), generator=g).item()
    else:
        obj_idx = rule_object[c, rel_idx].item()
    return subj_idx, rel_idx, obj_idx


def key_of(subj_idx, rel_idx, entity_vecs, relation_vecs):
    return torch.nn.functional.normalize(entity_vecs[subj_idx] + relation_vecs[rel_idx], dim=0)


def decode(vec, object_vecs):
    sims = object_vecs @ vec
    return int(sims.argmax().item())


def run():
    entity_vecs, entity_category, relation_vecs, object_vecs, rule_object = build_world()
    g = torch.Generator().manual_seed(7)

    n_entities = entity_vecs.shape[0]
    # Держим 20% сущностей каждой категории ЦЕЛИКОМ вне обучения - для zero-shot теста
    all_idx = list(range(n_entities))
    random.shuffle(all_idx)
    holdout_idx = set(all_idx[: int(0.2 * n_entities)])
    train_idx = [i for i in all_idx if i not in holdout_idx]

    DAYS = 10
    FACTS_PER_DAY = 80

    hippo = SDRHippocampus(dim=DIM, sdr_dim=1024, sparsity=0.06, beta=0.9, seed=1)
    cortex = PredictiveCodingNet([DIM, 64, DIM], relax_steps=60, relax_lr=0.05, weight_lr=0.006,
                                  seed=2, adam=True, weight_decay=0.03)

    exception_facts = []
    seen_train_keys = set()
    replay_archive = []  # накопленный архив дней - без него "сон" не реплеит прошлое вообще

    for day in range(1, DAYS + 1):
        day_facts = []
        while len(day_facts) < FACTS_PER_DAY:
            s = train_idx[torch.randint(0, len(train_idx), (1,), generator=g).item()]
            r = torch.randint(0, N_RELATIONS, (1,), generator=g).item()
            if (s, r) in seen_train_keys:
                continue
            seen_train_keys.add((s, r))
            is_exception = torch.rand(1, generator=g).item() < EXCEPTION_RATE
            s_, r_, o_ = make_fact(s, r, entity_category, rule_object, exception=is_exception, g=g)
            day_facts.append((s_, r_, o_, is_exception))
            if is_exception:
                exception_facts.append((s_, r_, o_))

        # WAKE: гиппокамп пишет всё, что видит (как и должен - zero backward)
        for s, r, o, is_exc in day_facts:
            key = key_of(s, r, entity_vecs, relation_vecs)
            hippo.write(key, object_vecs[o])

        # SLEEP: кора обучается PC-релаксацией на реплее дня + ВСЕГО архива
        # прошлых дней (полный rehearsal, не только сегодня - иначе кора
        # забывает предыдущие дни и не может накопить схему). backward() нигде.
        replay_archive.extend([(s, r, o) for s, r, o, _ in day_facts])
        Xb = torch.stack([key_of(s, r, entity_vecs, relation_vecs) for s, r, o in replay_archive])
        Yb = torch.stack([object_vecs[o] for s, r, o in replay_archive])
        for _ in range(25):
            cortex.train_step(Xb, Yb)

        print(f"День {day:2d}/{DAYS}: {len(day_facts)} фактов ({sum(f[3] for f in day_facts)} исключений), "
              f"архив реплея={len(replay_archive)}")

    # --- ТЕСТ (a): recall исключений - гиппокамп должен выигрывать ---
    exc_hippo_correct = exc_cortex_correct = 0
    for s, r, o in exception_facts:
        key = key_of(s, r, entity_vecs, relation_vecs)
        h_pred = decode(hippo.read(key), object_vecs)
        c_pred = decode(cortex.forward_pass(key.unsqueeze(0)).squeeze(0), object_vecs)
        exc_hippo_correct += (h_pred == o)
        exc_cortex_correct += (c_pred == o)
    exc_hippo_acc = exc_hippo_correct / len(exception_facts)
    exc_cortex_acc = exc_cortex_correct / len(exception_facts)

    # --- ТЕСТ (b): zero-shot на НИКОГДА не виденных сущностях той же категории ---
    zs_hippo_correct = zs_cortex_correct = 0
    zs_facts = []
    holdout_list = list(holdout_idx)
    for s in holdout_list:
        for r in range(N_RELATIONS):
            s_, r_, o_ = make_fact(s, r, entity_category, rule_object, exception=False)
            zs_facts.append((s_, r_, o_))
    for s, r, o in zs_facts:
        key = key_of(s, r, entity_vecs, relation_vecs)
        h_pred = decode(hippo.read(key), object_vecs)
        c_pred = decode(cortex.forward_pass(key.unsqueeze(0)).squeeze(0), object_vecs)
        zs_hippo_correct += (h_pred == o)
        zs_cortex_correct += (c_pred == o)
    zs_hippo_acc = zs_hippo_correct / len(zs_facts)
    zs_cortex_acc = zs_cortex_correct / len(zs_facts)

    chance = 1.0 / object_vecs.shape[0]

    print("=" * 70)
    print(f"(a) Recall ИСКЛЮЧЕНИЙ (N={len(exception_facts)}, произвольные, вне схемы):")
    print(f"    Гиппокамп: {exc_hippo_acc*100:5.1f}%   Кора: {exc_cortex_acc*100:5.1f}%")
    print(f"(b) ZERO-SHOT на новых сущностях известной категории (N={len(zs_facts)}, никогда не виденных):")
    print(f"    Гиппокамп: {zs_hippo_acc*100:5.1f}%   Кора: {zs_cortex_acc*100:5.1f}%   (случайный шанс: {chance*100:.1f}%)")
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Recall исключений\n(вне схемы)", "Zero-shot на новых\nсущностях (по схеме)"]
    hippo_vals = [exc_hippo_acc * 100, zs_hippo_acc * 100]
    cortex_vals = [exc_cortex_acc * 100, zs_cortex_acc * 100]
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width/2 for i in x], hippo_vals, width, label="Гиппокамп (lookup)", color="steelblue")
    ax.bar([i + width/2 for i in x], cortex_vals, width, label="Кора (PC-релаксация, обобщение)", color="darkorange")
    ax.axhline(chance * 100, color="gray", linestyle=":", label=f"Случайный шанс ({chance*100:.1f}%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Гиппокамп vs Кора: разделение труда (исключения vs обобщение схемы)")
    ax.legend()
    ax.grid(True, axis="y")
    plt.tight_layout()
    path = os.path.join(plots_dir, "sleep_generalization_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")

    return exc_hippo_acc, exc_cortex_acc, zs_hippo_acc, zs_cortex_acc


if __name__ == "__main__":
    run()

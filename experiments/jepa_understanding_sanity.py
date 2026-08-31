"""
Проверка M4 (ROADMAP.md): "понимание" (классификация типа реплики) должно
возникать как побочный продукт представлений, полезных для самонадзорного
предсказания - а не быть зашитым if/elif, как было в v1's milestone3_dialogue.py.

Протокол:
1. Самонадзорное предобучение (JEPA-стиль: предсказать замаскированный
   токен по остальным) БЕЗ единой метки интента - PC-релаксация, zero backward.
2. Линейный пробник поверх замороженных представлений - обучен на МАЛОМ
   числе размеченных примеров из ВИДЕННЫХ шаблонов.
3. Честный held-out: 4-й шаблон каждой категории НИКОГДА не встречается
   ни в самонадзорном предобучении, ни в обучении пробника.
4. Baseline: hardcoded-парсер (позиционные правила, как в v1), откалиброванный
   ТОЛЬКО на виденных шаблонах - структурно должен спотыкаться на новой
   формулировке, если она не совпадает по паттерну.
"""
import sys, os, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from core.predictive_coding import PredictiveCodingNet

torch.manual_seed(42)
random.seed(42)

EMBED_DIM = 32
HIDDEN_DIM = 64

VOCAB = ["<pad>", "alex", "bob", "clara", "coffee", "tea", "chess", "music",
         "likes", "is", "saw", "with", "really", "loves", "i",
         "what", "does", "like", "tell", "me", "about", "can", "you", "say",
         "how", "are", "ok", "whats", "up", "do", "feel", "today"]
VOCAB_IDX = {w: i for i, w in enumerate(VOCAB)}
ENTITIES = ["alex", "bob", "clara"]
OBJECTS = ["coffee", "tea", "chess", "music"]

# (шаблон с {ent}/{obj} слотами, категория). Последний в каждом списке - held-out.
TEMPLATES = {
    # held-out (4-й) шаблон использует ТОЛЬКО слова, встречавшиеся хоть где-то
    # в T1-3 (любой категории) - это честный тест обобщения на НОВЫЙ ПОРЯДОК/
    # комбинацию известных слов, а не на обработку невиданных слов вообще
    # (первая версия по ошибке вводила новые слова "loves"/"can"/"say" - это
    # тестировало OOV-обработку, не generalization формулировки).
    "FACT": [
        "{ent} likes {obj}",
        "{ent} is {obj}",
        "i saw {ent} with {obj}",
        "{ent} is with {obj}",  # held-out: новая комбинация is+with
    ],
    "QUESTION": [
        "what does {ent} like",
        "does {ent} like {obj}",
        "tell me about {ent}",
        "tell me what {ent} like",  # held-out: новая комбинация tell+me+what+like
    ],
    "EMOTION": [
        "how are you",
        "are you ok",
        "whats up",
        "ok how are you",  # held-out: новый порядок известных слов
    ],
}
CATEGORIES = list(TEMPLATES.keys())


def fill(template, g):
    ent = ENTITIES[torch.randint(0, len(ENTITIES), (1,), generator=g).item()]
    obj = OBJECTS[torch.randint(0, len(OBJECTS), (1,), generator=g).item()]
    return template.format(ent=ent, obj=obj).split()


def build_embeddings(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(len(VOCAB), EMBED_DIM, generator=g), dim=-1)


def phrase_vec(tokens, embed):
    idx = torch.tensor([VOCAB_IDX[t] for t in tokens])
    return embed[idx].sum(dim=0)


def gen_examples(split, n_per_template, embed, g):
    """split: 'train' (шаблоны 0..2) или 'heldout' (шаблон 3)."""
    examples = []
    for cat in CATEGORIES:
        templates = TEMPLATES[cat][:3] if split == "train" else TEMPLATES[cat][3:]
        for tmpl in templates:
            for _ in range(n_per_template):
                tokens = fill(tmpl, g)
                examples.append((tokens, cat))
    return examples


def run():
    embed = build_embeddings()
    g = torch.Generator().manual_seed(7)

    # === 1. Самонадзорное предобучение: предсказать замаскированный токен ===
    # БЕЗ меток интента - чисто self-supervised, PC-релаксация, zero backward.
    pretrain_examples = gen_examples("train", n_per_template=40, embed=embed, g=g)
    ssl_inputs, ssl_targets = [], []
    for tokens, _cat in pretrain_examples:
        if len(tokens) < 2:
            continue
        mask_pos = torch.randint(0, len(tokens), (1,), generator=g).item()
        context = tokens[:mask_pos] + tokens[mask_pos + 1:]
        target_tok = tokens[mask_pos]
        ssl_inputs.append(phrase_vec(context, embed))
        ssl_targets.append(embed[VOCAB_IDX[target_tok]])
    Xssl = torch.stack(ssl_inputs)
    Yssl = torch.stack(ssl_targets)

    ssl_net = PredictiveCodingNet([EMBED_DIM, HIDDEN_DIM, EMBED_DIM], relax_steps=60, relax_lr=0.05,
                                   weight_lr=0.008, seed=2, adam=True, weight_decay=0.02)
    print("Самонадзорное предобучение (предсказание замаскированного токена)...")
    for step in range(200):
        ssl_net.train_step(Xssl, Yssl)

    def hidden_rep(tokens):
        """Представление фразы - активация СКРЫТОГО слоя самонадзорной сети
        на ПОЛНОЙ (немаскированной) фразе. Это и есть выученное "понимание"."""
        v = phrase_vec(tokens, embed).unsqueeze(0)
        z1 = v @ ssl_net.W[0].T + ssl_net.b[0]
        return torch.tanh(z1).squeeze(0)

    # === 2. Линейный пробник: МАЛО размеченных примеров, только виденные шаблоны ===
    probe_examples = gen_examples("train", n_per_template=8, embed=embed, g=g)
    Xp = torch.stack([hidden_rep(t) for t, _ in probe_examples])
    Yp = torch.zeros(len(probe_examples), len(CATEGORIES))
    for i, (_, cat) in enumerate(probe_examples):
        Yp[i, CATEGORIES.index(cat)] = 1.0

    probe = PredictiveCodingNet([HIDDEN_DIM, len(CATEGORIES)], relax_steps=40, relax_lr=0.08,
                                 weight_lr=0.02, seed=3, adam=True, weight_decay=0.02)
    print("Обучение линейного пробника (мало примеров, только виденные шаблоны)...")
    for step in range(150):
        probe.train_step(Xp, Yp)

    def probe_predict(tokens):
        rep = hidden_rep(tokens).unsqueeze(0)
        logits = probe.forward_pass(rep).squeeze(0)
        return CATEGORIES[int(logits.argmax().item())]

    # === 3. Hardcoded-парсер (позиционные правила, откалиброванные ТОЛЬКО
    #        на виденных шаблонах - как в v1's milestone3_dialogue.py) ===
    def hardcoded_predict(tokens):
        if tokens[0] in ["what", "does", "tell"]:
            return "QUESTION"
        if tokens[0] in ["how", "are", "whats"]:
            return "EMOTION"
        if len(tokens) >= 2 and (tokens[1] in ["likes", "is"] or tokens[0] == "i"):
            return "FACT"
        return "FACT"  # default по большинству классов в train

    # === 4. ЧЕСТНЫЙ ТЕСТ: held-out шаблоны, никогда не виденные ни в SSL, ни в пробнике ===
    test_examples = gen_examples("heldout", n_per_template=50, embed=embed, g=g)

    probe_correct = sum(probe_predict(t) == cat for t, cat in test_examples)
    hardcoded_correct = sum(hardcoded_predict(t) == cat for t, cat in test_examples)
    n = len(test_examples)
    chance = 1.0 / len(CATEGORIES)

    print("=" * 70)
    print(f"ЧЕСТНЫЙ ТЕСТ на held-out шаблонах (N={n}, никогда не виденных):")
    print(f"  Пробник (JEPA-представления):  {100*probe_correct/n:5.1f}%")
    print(f"  Hardcoded-парсер (if/elif):    {100*hardcoded_correct/n:5.1f}%")
    print(f"  Случайный шанс:                {100*chance:5.1f}%")
    print("=" * 70)

    # Разбивка по категориям для честности - не прячем, если где-то хуже
    print("\nПо категориям (held-out):")
    for cat in CATEGORIES:
        cat_examples = [(t, c) for t, c in test_examples if c == cat]
        p_acc = sum(probe_predict(t) == c for t, c in cat_examples) / len(cat_examples)
        h_acc = sum(hardcoded_predict(t) == c for t, c in cat_examples) / len(cat_examples)
        print(f"  {cat:10s} (шаблон: '{TEMPLATES[cat][3]}'): пробник={p_acc*100:5.1f}%  hardcoded={h_acc*100:5.1f}%")

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = ["Пробник\n(JEPA-представления)", "Hardcoded\n(if/elif)", "Случайный шанс"]
    vals = [100*probe_correct/n, 100*hardcoded_correct/n, 100*chance]
    ax.bar(methods, vals, color=["darkorange", "steelblue", "gray"])
    ax.set_ylabel("Accuracy на held-out шаблонах (%)")
    ax.set_title("Понимание из предсказания (JEPA) vs зашитый парсер - на НОВЫХ формулировках")
    ax.grid(True, axis="y")
    plt.tight_layout()
    path = os.path.join(plots_dir, "jepa_understanding_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"\nPlot saved: {path}")

    return probe_correct / n, hardcoded_correct / n


if __name__ == "__main__":
    run()

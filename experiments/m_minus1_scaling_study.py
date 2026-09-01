"""
M(-1) из ROADMAP.md — единственный давно отложенный, но фундаментальный
пункт: закрывается ли разрыв качества PC-релаксации (zero backward) с
backprop при росте бюджета обучения, на РЕАЛЬНОЙ задаче (char-level
language modeling на реальном тексте), не на игрушечной спирали?

Датасет: tiny-shakespeare (архив v1, 1.1MB реального текста - тот же
корпус, что и в первом (недообученном!) сравнении v1's grand_benchmark.py).

Протокол: идентичная архитектура (context_dim -> hidden -> vocab_size),
идентичный MSE loss (one-hot target), идентичные фиксированные embeddings
(ни PC, ни BP не учат embedding, только предсказывающую MLP - изолирует
сравнение чисто до "локальное правило vs backprop", как и в самом первом
нашем тесте). Несколько БЮДЖЕТОВ обучения (не одна точка!), чтобы увидеть,
сужается ли разрыв, держится константным или растёт.
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from core.predictive_coding import PredictiveCodingNet

torch.manual_seed(42)

CORPUS_PATH = "/root/archive/nca_research_v1/data/input.txt"
CONTEXT_LEN = 16
EMBED_DIM = 16
HIDDEN = 128
BATCH_SIZE = 64


def load_corpus():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, len(chars)


def get_batch(data, embed, g, batch_size=BATCH_SIZE):
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (batch_size,), generator=g)
    ctx_idx = torch.stack([data[i:i + CONTEXT_LEN] for i in ix])  # (B, CONTEXT_LEN)
    target_idx = torch.stack([data[i + CONTEXT_LEN] for i in ix])  # (B,)
    ctx_vec = embed(ctx_idx).view(batch_size, -1)  # (B, CONTEXT_LEN*EMBED_DIM)
    return ctx_vec, target_idx


class BPNet(nn.Module):
    def __init__(self, in_dim, hidden, vocab_size):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


def eval_accuracy_pc(net, data, embed, g, n=2000):
    ctx_vec, target_idx = get_batch(data, embed, g, batch_size=n)
    logits = net.forward_pass(ctx_vec)
    return (logits.argmax(dim=1) == target_idx).float().mean().item()


def eval_accuracy_bp(net, data, embed, g, n=2000):
    ctx_vec, target_idx = get_batch(data, embed, g, batch_size=n)
    with torch.no_grad():
        logits = net(ctx_vec)
    return (logits.argmax(dim=1) == target_idx).float().mean().item()


def run():
    data, vocab_size = load_corpus()
    n_train = int(0.9 * len(data))
    train_data, test_data = data[:n_train], data[n_train:]
    print(f"Корпус: {len(data)} символов, vocab_size={vocab_size}, train={n_train}, test={len(data)-n_train}")

    g_embed = torch.Generator().manual_seed(1)
    embed = nn.Embedding(vocab_size, EMBED_DIM)
    with torch.no_grad():
        embed.weight.normal_(0, 1.0, generator=g_embed)
    for p in embed.parameters():
        p.requires_grad_(False)  # embeddings фиксированы для ОБЕИХ моделей - честная изоляция

    in_dim = CONTEXT_LEN * EMBED_DIM
    BUDGETS = [100, 300, 1000, 3000, 8000]

    pc_results, bp_results = [], []
    g_data = torch.Generator().manual_seed(2)

    for budget in BUDGETS:
        # --- PC: zero backward, свежая сеть на каждый бюджет ---
        pc_net = PredictiveCodingNet([in_dim, HIDDEN, vocab_size], relax_steps=40, relax_lr=0.05,
                                      weight_lr=0.006, seed=3, adam=True, weight_decay=0.02)
        t0 = time.time()
        for step in range(budget):
            xb, yb_idx = get_batch(train_data, embed, g_data)
            yb = torch.zeros(BATCH_SIZE, vocab_size)
            yb.scatter_(1, yb_idx.unsqueeze(1), 1.0)
            pc_net.train_step(xb, yb)
        pc_time = time.time() - t0
        pc_acc = eval_accuracy_pc(pc_net, test_data, embed, torch.Generator().manual_seed(999))
        pc_results.append((budget, pc_acc, pc_time))
        print(f"PC  budget={budget:5d}  test_acc={pc_acc*100:5.1f}%  time={pc_time:6.1f}s")

        # --- BP: тот же budget (число шагов), backprop+Adam ---
        bp_net = BPNet(in_dim, HIDDEN, vocab_size)
        opt = torch.optim.Adam(bp_net.parameters(), lr=0.003)
        t0 = time.time()
        for step in range(budget):
            xb, yb_idx = get_batch(train_data, embed, g_data)
            opt.zero_grad()
            logits = bp_net(xb)
            loss = nn.functional.cross_entropy(logits, yb_idx)
            loss.backward()
            opt.step()
        bp_time = time.time() - t0
        bp_acc = eval_accuracy_bp(bp_net, test_data, embed, torch.Generator().manual_seed(999))
        bp_results.append((budget, bp_acc, bp_time))
        print(f"BP  budget={budget:5d}  test_acc={bp_acc*100:5.1f}%  time={bp_time:6.1f}s")
        print(f"    Разрыв (BP-PC): {(bp_acc-pc_acc)*100:+.1f} п.п.")
        print("-" * 60)

    print("=" * 70)
    print("Разрыв (BP - PC accuracy) по бюджетам:")
    for (b, pc_a, _), (_, bp_a, _) in zip(pc_results, bp_results):
        print(f"  budget={b:5d}  разрыв={{:+.1f}} п.п.".format((bp_a - pc_a) * 100))
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([b for b, a, t in pc_results], [a * 100 for b, a, t in pc_results], "g-o", label="PC (zero backward)", linewidth=2)
    ax.plot([b for b, a, t in bp_results], [a * 100 for b, a, t in bp_results], "r--s", label="Backprop", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Бюджет обучения (число шагов, log scale)")
    ax.set_ylabel("Test accuracy предсказания следующего символа (%)")
    ax.set_title("M(-1): закрывается ли разрыв PC vs backprop с ростом бюджета? (реальный текст, tiny-shakespeare)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(plots_dir, "m_minus1_scaling_study.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")

    return pc_results, bp_results


if __name__ == "__main__":
    run()

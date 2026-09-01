"""
Продолжение pc_ceiling_vs_network_size.py (был запущен ТОЛЬКО на Colab GPU,
до hidden=4096) - в этой сессии Colab GPU-квота недоступна очень долго
(много часов подряд), а вопрос "держится ли потолок PC на бОльших сетях"
из README явно помечен как приоритет. CPU-версия того же протокола (тот же
корпус tinyshakespeare - `data/text/corpus.txt`, скачан напрямую, тот же
размер 1115394 байт, что канонический файл, вероятно тот же самый, что
использовался на Colab) с СОКРАЩЁННЫМ бюджетом (CPU на порядки медленнее
GPU для этой сети) - не прямая замена Colab-теста, честно другой масштаб,
но новая точка данных на промежуточном hidden, недоступная раньше.
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from core.predictive_coding import PredictiveCodingNet

torch.manual_seed(42)

CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/text/corpus.txt")
CONTEXT_LEN = 24
EMBED_DIM = 32
BATCH_SIZE = 128


def load_corpus():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, len(chars)


def get_batch(data, embed, g, batch_size=BATCH_SIZE):
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (batch_size,), generator=g).tolist()
    ctx_idx = torch.stack([data[i:i + CONTEXT_LEN] for i in ix])
    target_idx = torch.stack([data[i + CONTEXT_LEN] for i in ix])
    ctx_vec = embed(ctx_idx).view(batch_size, -1)
    return ctx_vec, target_idx


class BPNet(nn.Module):
    def __init__(self, in_dim, hidden, vocab_size):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


def eval_acc_pc(net, data, embed, n=3000):
    g = torch.Generator().manual_seed(999)
    ctx_vec, target_idx = get_batch(data, embed, g, batch_size=n)
    logits = net.forward_pass(ctx_vec)
    return (logits.argmax(dim=1) == target_idx).float().mean().item()


def eval_acc_bp(net, data, embed, n=3000):
    g = torch.Generator().manual_seed(999)
    ctx_vec, target_idx = get_batch(data, embed, g, batch_size=n)
    with torch.no_grad():
        logits = net(ctx_vec)
    return (logits.argmax(dim=1) == target_idx).float().mean().item()


def train_pc(in_dim, hidden, vocab_size, train_data, embed, budget, relax_steps, relax_lr,
             weight_lr, gate_floor, weight_decay, seed=3):
    net = PredictiveCodingNet([in_dim, hidden, vocab_size], relax_steps=relax_steps, relax_lr=relax_lr,
                              weight_lr=weight_lr, seed=seed, adam=True, weight_decay=weight_decay,
                              gate_floor=gate_floor)
    g = torch.Generator().manual_seed(2)
    t0 = time.time()
    for step in range(budget):
        xb, yb_idx = get_batch(train_data, embed, g, batch_size=BATCH_SIZE)
        yb = torch.zeros(BATCH_SIZE, vocab_size)
        yb.scatter_(1, yb_idx.unsqueeze(1), 1.0)
        net.train_step(xb, yb)
    return net, time.time() - t0


def run():
    data, vocab_size = load_corpus()
    print(f"Корпус: {len(data)} символов, vocab_size={vocab_size}")
    n_train = int(0.9 * len(data))
    train_data, test_data = data[:n_train], data[n_train:]
    embed = nn.Embedding(vocab_size, EMBED_DIM)
    with torch.no_grad():
        embed.weight.normal_(0, 1.0, generator=torch.Generator().manual_seed(1))
    for p in embed.parameters():
        p.requires_grad_(False)
    in_dim = CONTEXT_LEN * EMBED_DIM

    best = dict(relax_steps=20, relax_lr=0.03, weight_lr=0.003, gate_floor=0.05, weight_decay=0.02)
    BUDGET = 2000  # сокращено с 20000 (Colab) - CPU не потянет тот же бюджет за разумное время
    HIDDENS = [512, 768, 1024]  # 768 - промежуточная точка, никогда не тестировалась даже на Colab

    print("=" * 70)
    print(f"CPU follow-up (Colab недоступен): budget={BUDGET} (сокращено с 20000 на Colab)")
    print(f"Гиперпараметры (те же, что нашли для hidden=512 на Colab): {best}")
    print("=" * 70)

    results = {}
    for hidden in HIDDENS:
        t0 = time.time()
        net, t = train_pc(in_dim, hidden, vocab_size, train_data, embed, BUDGET, **best)
        acc = eval_acc_pc(net, test_data, embed)
        results[hidden] = acc
        print(f"  hidden={hidden:5d}  PC_acc={acc*100:5.1f}%  time={t:6.1f}s")

    print("\nBP-референс на том же диапазоне hidden, budget...")
    bp_results = {}
    for hidden in HIDDENS:
        bp_net = BPNet(in_dim, hidden, vocab_size)
        opt = torch.optim.Adam(bp_net.parameters(), lr=0.003)
        g = torch.Generator().manual_seed(2)
        t0 = time.time()
        for step in range(BUDGET):
            xb, yb_idx = get_batch(train_data, embed, g)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(bp_net(xb), yb_idx)
            loss.backward()
            opt.step()
        acc = eval_acc_bp(bp_net, test_data, embed)
        bp_results[hidden] = acc
        print(f"  hidden={hidden:5d}  BP_acc={acc*100:5.1f}%  time={time.time()-t0:6.1f}s")

    print("\n" + "=" * 70)
    print(f"ИТОГ (budget={BUDGET}, CPU, сокращённый масштаб относительно Colab-теста):")
    for hidden in HIDDENS:
        gap = (bp_results[hidden] - results[hidden]) * 100
        print(f"  hidden={hidden:5d}  PC={results[hidden]*100:5.1f}%  BP={bp_results[hidden]*100:5.1f}%  разрыв={gap:+.1f}п.п.")
    print("=" * 70)

    return results, bp_results


if __name__ == "__main__":
    run()

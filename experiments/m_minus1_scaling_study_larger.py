import sys, os, time
sys.path.append("/content")
import torch
import torch.nn as nn
from core.predictive_coding import PredictiveCodingNet

torch.manual_seed(42)

CORPUS_PATH = "/content/data/input.txt"
CONTEXT_LEN = 24
EMBED_DIM = 32
HIDDEN = 512  # 4x больше локального теста (было 128)
BATCH_SIZE = 128
DEV = "cuda"


def pc_to_cuda(net):
    """Переносим веса уже сконструированного (CPU) PredictiveCodingNet на GPU
    вручную, извне - без monkeypatch torch.Generator (первая попытка сломала
    несвязанный код: torch.optim.Adam лениво импортирует torch._dynamo, где
    generator: torch.Generator | None ломается, если Generator - не класс)."""
    net.W = [w.to(DEV) for w in net.W]
    net.b = [b.to(DEV) for b in net.b]
    if net.adam:
        net.mW = [m.to(DEV) for m in net.mW]
        net.vW = [v.to(DEV) for v in net.vW]
        net.mb = [m.to(DEV) for m in net.mb]
        net.vb = [v.to(DEV) for v in net.vb]
    return net


def load_corpus():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=DEV)
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


def train_pc(in_dim, vocab_size, train_data, embed, budget, relax_steps, relax_lr,
             weight_lr, gate_floor, weight_decay, seed=3):
    net = PredictiveCodingNet([in_dim, HIDDEN, vocab_size], relax_steps=relax_steps, relax_lr=relax_lr,
                              weight_lr=weight_lr, seed=seed, adam=True, weight_decay=weight_decay,
                              gate_floor=gate_floor)
    pc_to_cuda(net)
    g = torch.Generator().manual_seed(2)
    t0 = time.time()
    for step in range(budget):
        xb, yb_idx = get_batch(train_data, embed, g)
        yb = torch.zeros(BATCH_SIZE, vocab_size, device=DEV)
        yb.scatter_(1, yb_idx.unsqueeze(1), 1.0)
        net.train_step(xb, yb)
    torch.cuda.synchronize()
    return net, time.time() - t0


def run():
    data, vocab_size = load_corpus()
    n_train = int(0.9 * len(data))
    train_data, test_data = data[:n_train], data[n_train:]
    print(f"Корпус: {len(data)} симв, vocab={vocab_size}, HIDDEN={HIDDEN}, CONTEXT={CONTEXT_LEN}")

    embed = nn.Embedding(vocab_size, EMBED_DIM)
    with torch.no_grad():
        embed.weight.normal_(0, 1.0, generator=torch.Generator().manual_seed(1))
    embed = embed.to(DEV)
    for p in embed.parameters():
        p.requires_grad_(False)
    in_dim = CONTEXT_LEN * EMBED_DIM

    # Лучшая конфигурация, найденная на Фазе 1 (перебор гиперпараметров,
    # успешно завершён до крэша Фазы 2 из-за monkeypatch-бага):
    # relax_steps=20 (не 40 - меньше оказалось лучше), relax_lr=0.03,
    # weight_lr=0.003, gate_floor=0.05 (без разницы 0.05/0.2/0.5 - не было
    # узким местом на этом бюджете).
    best = dict(relax_steps=20, relax_lr=0.03, weight_lr=0.003, gate_floor=0.05, weight_decay=0.02)
    print(f"Используем найденную на Фазе 1 конфигурацию: {best}")

    print("=" * 70)
    print("ФАЗА 2: полный sweep бюджетов на большей сети (HIDDEN=512) с лучшим конфигом")
    print("=" * 70)
    BUDGETS = [100, 300, 1000, 3000, 8000, 20000, 40000]
    pc_results, bp_results = [], []
    for budget in BUDGETS:
        net, t = train_pc(in_dim, vocab_size, train_data, embed, budget, **best)
        acc = eval_acc_pc(net, test_data, embed)
        pc_results.append((budget, acc, t))
        print(f"PC  budget={budget:6d}  acc={acc*100:5.1f}%  time={t:6.1f}s")

        bp_net = BPNet(in_dim, HIDDEN, vocab_size).to(DEV)
        opt = torch.optim.Adam(bp_net.parameters(), lr=0.003)
        g = torch.Generator().manual_seed(2)
        t0 = time.time()
        for step in range(budget):
            xb, yb_idx = get_batch(train_data, embed, g)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(bp_net(xb), yb_idx)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        bp_t = time.time() - t0
        bp_acc = eval_acc_bp(bp_net, test_data, embed)
        bp_results.append((budget, bp_acc, bp_t))
        print(f"BP  budget={budget:6d}  acc={bp_acc*100:5.1f}%  time={bp_t:6.1f}s")
        print(f"    Разрыв: {(bp_acc-acc)*100:+.1f} п.п.")
        print("-" * 60)

    print("=" * 70)
    print("ИТОГ (HIDDEN=512, лучшие найденные гиперпараметры PC):")
    for (b, pc_a, _), (_, bp_a, _) in zip(pc_results, bp_results):
        print(f"  budget={b:6d}  PC={pc_a*100:5.1f}%  BP={bp_a*100:5.1f}%  разрыв={(bp_a-pc_a)*100:+.1f} п.п.")
    print("=" * 70)


if __name__ == "__main__":
    run()

"""
Симметричная перепроверка M(-1): PC-гиперпараметры уже перебраны (найдено
relax_steps=20, relax_lr=0.03, weight_lr=0.003, gate_floor=0.05). Теперь
перебираем lr у BP С ТОЙ ЖЕ СТРОГОСТЬЮ на hidden=512, прежде чем сравнивать
полную кривую бюджетов - иначе сравнение нечестное (найдено на hidden=4096:
неправильный lr давал BP 29.4%, правильный - 44.6%).
"""
import sys, os, time
sys.path.append("/content")
import torch
import torch.nn as nn
from core.predictive_coding import PredictiveCodingNet

torch.manual_seed(42)

CORPUS_PATH = "/content/data/input.txt"
CONTEXT_LEN = 24
EMBED_DIM = 32
HIDDEN = 512
BATCH_SIZE = 128
DEV = "cuda"


def pc_to_cuda(net):
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


def train_bp(in_dim, hidden, vocab_size, train_data, embed, budget, lr):
    net = BPNet(in_dim, hidden, vocab_size).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(2)
    t0 = time.time()
    for step in range(budget):
        xb, yb_idx = get_batch(train_data, embed, g)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(xb), yb_idx)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return net, time.time() - t0


def train_pc(in_dim, hidden, vocab_size, train_data, embed, budget, relax_steps, relax_lr,
             weight_lr, gate_floor, weight_decay, seed=3):
    net = PredictiveCodingNet([in_dim, hidden, vocab_size], relax_steps=relax_steps, relax_lr=relax_lr,
                              weight_lr=weight_lr, seed=seed, adam=True, weight_decay=weight_decay,
                              gate_floor=gate_floor)
    pc_to_cuda(net)
    g = torch.Generator().manual_seed(2)
    t0 = time.time()
    for step in range(budget):
        xb, yb_idx = get_batch(train_data, embed, g, batch_size=BATCH_SIZE)
        yb = torch.zeros(BATCH_SIZE, vocab_size, device=DEV)
        yb.scatter_(1, yb_idx.unsqueeze(1), 1.0)
        net.train_step(xb, yb)
    torch.cuda.synchronize()
    return net, time.time() - t0


def run():
    data, vocab_size = load_corpus()
    n_train = int(0.9 * len(data))
    train_data, test_data = data[:n_train], data[n_train:]
    embed = nn.Embedding(vocab_size, EMBED_DIM)
    with torch.no_grad():
        embed.weight.normal_(0, 1.0, generator=torch.Generator().manual_seed(1))
    embed = embed.to(DEV)
    for p in embed.parameters():
        p.requires_grad_(False)
    in_dim = CONTEXT_LEN * EMBED_DIM

    print("=" * 70)
    print("ФАЗА 1: lr-sweep для BP на hidden=512 (budget=20000) - та же строгость, что для PC")
    print("=" * 70)
    bp_lr_results = []
    for lr in [0.0003, 0.001, 0.003, 0.01]:
        net, t = train_bp(in_dim, HIDDEN, vocab_size, train_data, embed, 20000, lr)
        acc = eval_acc_bp(net, test_data, embed)
        bp_lr_results.append((lr, acc))
        print(f"  BP lr={lr:.4f}  acc={acc*100:5.1f}%  time={t:.1f}s")
    best_bp_lr = max(bp_lr_results, key=lambda r: r[1])[0]
    print(f"  -> лучший BP lr на hidden=512: {best_bp_lr}")

    pc_best = dict(relax_steps=20, relax_lr=0.03, weight_lr=0.003, gate_floor=0.05, weight_decay=0.02)

    print("\n" + "=" * 70)
    print(f"ФАЗА 2: СИММЕТРИЧНОЕ сравнение (PC best config vs BP lr={best_bp_lr}), hidden=512")
    print("=" * 70)
    BUDGETS = [100, 300, 1000, 3000, 8000, 20000, 40000]
    for budget in BUDGETS:
        pc_net, pc_t = train_pc(in_dim, HIDDEN, vocab_size, train_data, embed, budget, **pc_best)
        pc_acc = eval_acc_pc(pc_net, test_data, embed)

        bp_net, bp_t = train_bp(in_dim, HIDDEN, vocab_size, train_data, embed, budget, best_bp_lr)
        bp_acc = eval_acc_bp(bp_net, test_data, embed)

        print(f"budget={budget:6d}  PC={pc_acc*100:5.1f}% ({pc_t:.1f}s)  "
              f"BP(lr={best_bp_lr})={bp_acc*100:5.1f}% ({bp_t:.1f}s)  "
              f"разрыв={(bp_acc-pc_acc)*100:+.1f} п.п.")


if __name__ == "__main__":
    run()

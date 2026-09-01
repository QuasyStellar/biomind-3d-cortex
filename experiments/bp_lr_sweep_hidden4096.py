import sys, time
sys.path.append("/content")
import torch, torch.nn as nn

DEV = "cuda"
CORPUS_PATH = "/content/data/input.txt"
CONTEXT_LEN = 24
EMBED_DIM = 32
BATCH_SIZE = 128


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


def eval_acc(net, data, embed, n=3000):
    g = torch.Generator().manual_seed(999)
    ctx_vec, target_idx = get_batch(data, embed, g, batch_size=n)
    with torch.no_grad():
        logits = net(ctx_vec)
    return (logits.argmax(dim=1) == target_idx).float().mean().item()


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

HIDDEN = 4096
BUDGET = 20000
print(f"BP @ hidden={HIDDEN}, budget={BUDGET}, перебор learning rate:")
for lr in [0.0003, 0.001, 0.003, 0.01]:
    bp_net = BPNet(in_dim, HIDDEN, vocab_size).to(DEV)
    opt = torch.optim.Adam(bp_net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(2)
    t0 = time.time()
    for step in range(BUDGET):
        xb, yb_idx = get_batch(train_data, embed, g)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(bp_net(xb), yb_idx)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    acc = eval_acc(bp_net, test_data, embed)
    print(f"  lr={lr:.4f}  acc={acc*100:5.1f}%  time={time.time()-t0:.1f}s")

"""
M9 (docs/ROADMAP.md): "Почему разрыв PC-vs-BP уже на реальных данных" -
проверяем гипотезу напрямую, а не только на бинарном MNIST-vs-синтетика.

Гипотеза: PC-релаксация эффективнее использует ЕСТЕСТВЕННУЮ статистическую/
пространственную структуру данных (корреляция соседних измерений входного
вектора), чем полностью случайные representations M(-1) - если так, сужение
разрыва должно быть систематической функцией СТЕПЕНИ структурированности
данных, не спецификой MNIST как таковой.

Протокол: тот же симметрично настроенный протокол, что в финальном M(-1)
(`m_minus1_symmetric_comparison.py`: CONTEXT_LEN=24, EMBED_DIM=32, HIDDEN=512,
budget=8000 - точка с честным разрывом 6.3 п.п. из полного sweep, не самая
шумная), но embedding-таблица символов синтетически размывается по оси
EMBED_DIM (не по оси словаря - размытие вдоль измерений ОДНОГО вектора,
напрямую аналогично тому, что смежные пиксели MNIST коррелированы вдоль
своих измерений) с растущим sigma - от sigma=0 (чистый M(-1) baseline,
i.i.d. случайные векторы) до sigma=8 (сильно сглаженные, структурированные
векторы). Метрика структурированности - явная, не "синтетика vs реальность"
бинарно: средняя корреляция Пирсона между СОСЕДНИМИ измерениями входного
вектора, усреднённая по измерениям и сэмплам - вычисляется ОДИНАКОВО для
embedding-векторов (эта работа) и для сырых пикселей MNIST (переиспользуем
уже готовый результат PC-vs-BP разрыва оттуда, ~1.79 п.п. по 9 прогонам).

Честная оговорка по протоколу: "2+ реальных датасета" из ROADMAP выполнено
частично - последовательность символов реальная (Shakespeare) во ВСЕЙ
sigma-развёртке, но embedding-representation синтетическая при sigma>=0;
MNIST - полностью реальные данные и representation. Это разные категории
"реальности", не скрываем эту разницу.
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.predictive_coding import PredictiveCodingNet

torch.manual_seed(42)

CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "text", "corpus.txt")
CONTEXT_LEN = 24
EMBED_DIM = 32
HIDDEN = 512
BATCH_SIZE = 128
BUDGET = 8000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIGMAS = [0.0, 1.0, 2.0, 4.0, 8.0]

PC_BEST = dict(relax_steps=20, relax_lr=0.03, weight_lr=0.003, gate_floor=0.05, weight_decay=0.02)


def pc_to_dev(net):
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


def build_embedding(vocab_size, embed_dim, seed, sigma):
    """Случайная embedding-таблица, синтетически размытая ВДОЛЬ embed_dim
    (каждый вектор символа сглаживается по своим собственным измерениям) -
    растущий sigma даёт растущую корреляцию СОСЕДНИХ измерений вектора,
    напрямую аналогично соседним пикселям изображения."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(vocab_size, embed_dim, generator=g)
    if sigma <= 0:
        return base.to(DEV)
    k = max(3, int(6 * sigma) | 1)
    xs = torch.arange(k, dtype=torch.float32) - k // 2
    kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    x = base.unsqueeze(1)  # (vocab_size, 1, embed_dim)
    pad = k // 2
    x_pad = F.pad(x, (pad, pad), mode="reflect")
    blurred = F.conv1d(x_pad, kernel).squeeze(1)  # (vocab_size, embed_dim)
    blurred = blurred / blurred.std() * base.std()
    return blurred.to(DEV)


def structuredness(X):
    """Средняя корреляция Пирсона между соседними измерениями входного
    вектора, усреднённая по измерениям - одна и та же формула для
    embedding-векторов и для сырых пикселей MNIST (независимо считается
    в mnist_structuredness_reference.py)."""
    Xz = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-7)
    prod = (Xz[:, :-1] * Xz[:, 1:]).mean(dim=0)
    return prod.mean().item()


def get_batch(data, embed_matrix, g, batch_size=BATCH_SIZE):
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (batch_size,), generator=g).tolist()
    ctx_idx = torch.stack([data[i:i + CONTEXT_LEN] for i in ix])
    target_idx = torch.stack([data[i + CONTEXT_LEN] for i in ix])
    ctx_vec = embed_matrix[ctx_idx].view(batch_size, -1)
    return ctx_vec, target_idx


class BPNet(nn.Module):
    def __init__(self, in_dim, hidden, vocab_size):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


def eval_acc_pc(net, data, embed_matrix, n=3000):
    g = torch.Generator().manual_seed(999)
    ctx_vec, target_idx = get_batch(data, embed_matrix, g, batch_size=n)
    logits = net.forward_pass(ctx_vec)
    return (logits.argmax(dim=1) == target_idx).float().mean().item()


def eval_acc_bp(net, data, embed_matrix, n=3000):
    g = torch.Generator().manual_seed(999)
    ctx_vec, target_idx = get_batch(data, embed_matrix, g, batch_size=n)
    with torch.no_grad():
        logits = net(ctx_vec)
    return (logits.argmax(dim=1) == target_idx).float().mean().item()


def train_bp(in_dim, hidden, vocab_size, train_data, embed_matrix, budget, lr):
    net = BPNet(in_dim, hidden, vocab_size).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(2)
    for step in range(budget):
        xb, yb_idx = get_batch(train_data, embed_matrix, g)
        opt.zero_grad()
        loss = F.cross_entropy(net(xb), yb_idx)
        loss.backward()
        opt.step()
    return net


def train_pc(in_dim, hidden, vocab_size, train_data, embed_matrix, budget):
    net = PredictiveCodingNet([in_dim, hidden, vocab_size], seed=3, adam=True, **PC_BEST)
    pc_to_dev(net)
    g = torch.Generator().manual_seed(2)
    for step in range(budget):
        xb, yb_idx = get_batch(train_data, embed_matrix, g, batch_size=BATCH_SIZE)
        yb = torch.zeros(BATCH_SIZE, vocab_size, device=DEV)
        yb.scatter_(1, yb_idx.unsqueeze(1), 1.0)
        net.train_step(xb, yb)
    return net


def run():
    data, vocab_size = load_corpus()
    n_train = int(0.9 * len(data))
    train_data, test_data = data[:n_train], data[n_train:]
    in_dim = CONTEXT_LEN * EMBED_DIM

    print("=" * 70)
    print(f"DEV={DEV}  vocab_size={vocab_size}  budget={BUDGET}  hidden={HIDDEN}")
    print("ФАЗА 1: BP lr-sweep на sigma=0 (тот же символьный масштаб, что M(-1))")
    embed0 = build_embedding(vocab_size, EMBED_DIM, seed=1, sigma=0.0)
    bp_lr_results = []
    for lr in [0.0001, 0.0003, 0.001, 0.003]:
        t0 = time.time()
        net = train_bp(in_dim, HIDDEN, vocab_size, train_data, embed0, BUDGET, lr)
        acc = eval_acc_bp(net, test_data, embed0)
        bp_lr_results.append((lr, acc))
        print(f"  BP lr={lr:.4f}  acc={acc*100:5.1f}%  ({time.time()-t0:.1f}s)")
    best_bp_lr = max(bp_lr_results, key=lambda r: r[1])[0]
    print(f"  -> лучший BP lr: {best_bp_lr}")

    print("\n" + "=" * 70)
    print(f"ФАЗА 2: sigma-развёртка (N_SIGMA={len(SIGMAS)}, budget={BUDGET}, "
          f"PC-конфиг фикс., BP lr={best_bp_lr} фикс.)")
    print("=" * 70)
    rows = []
    for sigma in SIGMAS:
        embed = build_embedding(vocab_size, EMBED_DIM, seed=1, sigma=sigma)
        struct = structuredness(embed)

        t0 = time.time()
        pc_net = train_pc(in_dim, HIDDEN, vocab_size, train_data, embed, BUDGET)
        pc_acc = eval_acc_pc(pc_net, test_data, embed)
        pc_t = time.time() - t0

        t0 = time.time()
        bp_net = train_bp(in_dim, HIDDEN, vocab_size, train_data, embed, BUDGET, best_bp_lr)
        bp_acc = eval_acc_bp(bp_net, test_data, embed)
        bp_t = time.time() - t0

        gap = (bp_acc - pc_acc) * 100
        rows.append((sigma, struct, pc_acc, bp_acc, gap))
        print(f"sigma={sigma:4.1f}  structuredness={struct:+.4f}  "
              f"PC={pc_acc*100:5.1f}%({pc_t:.0f}s)  BP={bp_acc*100:5.1f}%({bp_t:.0f}s)  "
              f"gap={gap:+.1f}п.п.")

    print("\n" + "=" * 70)
    print("ИТОГ (sigma, structuredness, gap п.п.):")
    for sigma, struct, pc_acc, bp_acc, gap in rows:
        print(f"  sigma={sigma:4.1f}  structuredness={struct:+.4f}  gap={gap:+.1f}")
    print("=" * 70)


if __name__ == "__main__":
    run()

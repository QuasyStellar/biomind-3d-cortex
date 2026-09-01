"""
M9, вторая попытка - по формулировке протокола, исправленной в этой же
сессии (docs/ROADMAP.md, "ПРЕДПОСЫЛКА ИСПРАВЛЕНА"). Первая попытка
(`m9_structuredness_vs_gap.py`) была построена на предпосылке "M(-1) -
уже готовая чистая синтетика" - неверно (M(-1) всегда был на реальном
тексте), и вдобавок её 1D-размытие вдоль EMBED_DIM схлопывало
эффективную размерность (rank) вместе с ростом "структурированности",
спутывая две разные вещи - см. её честную запись в VERIFICATION_LOG.

Здесь - ЗАНОВО построенная синтетика с нуля, той же архитектуры, что и
честный MNIST-тест (784->128->10, PC: weight_lr=0.01/relax_steps=20/
relax_lr=0.08/weight_decay=0.02, BP: lr=0.01, N_train=2000, 3 seed -
РОВНО тот же протокол, что в 9-прогонном MNIST sweep):

(а) N_CLASSES=10 независимых случайных "шаблонов" (784-dim, i.i.d.) +
    per-sample шум - НУЛЕВАЯ пространственная структура между соседними
    компонентами вектора (sigma=0).
(б) та же генерация, но результат reshape 28x28 и размывается 2D
    гауссовым ядром с растущим sigma - ВАЖНО, в отличие от первой
    попытки: дисперсия РЕНОРМАЛИЗУЕТСЯ после размытия к исходному
    уровню (контроль confound'а "размытие = потеря информации"), И
    печатается effective rank (SVD) на каждой точке - если и здесь
    rank катастрофически падает, честно останавливаемся и не делаем
    вывод по этой точке, а не игнорируем предупреждающий знак.

Единая метрика структурированности - средняя корреляция Пирсона между
соседними измерениями входного вектора (lag-1 автокорреляция), ТА ЖЕ
формула, что уже применена к реальным пикселям MNIST
(`m9_mnist_structuredness_reference.py`, structuredness=+0.5633) - чтобы
поставить на одну ось: (а)+(б) синтетика, (в) MNIST (+1.79+-0.75 п.п.,
9 прогонов) и M(-1) текст (5.9-9.9 п.п.) как два реальных якоря.
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.predictive_coding import PredictiveCodingNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 784
SIDE = 28
N_CLASSES = 10
N_TRAIN = 2000
N_TEST = 1000
NOISE_STD = 6.0  # откалибровано отдельно: даёт PC~73%/BP~93% на sigma=0 -
# не потолок (100/100, неинформативно) и не пол, здоровый динамический диапазон
SIGMAS = [0.0, 0.5, 1.0, 2.0, 4.0]
SEEDS = [1, 2, 3]
PC_KW = dict(relax_steps=20, relax_lr=0.08, weight_lr=0.01, weight_decay=0.02)
BP_LR = 0.01


def structuredness(X):
    Xz = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-7)
    prod = (Xz[:, :-1] * Xz[:, 1:]).mean(dim=0)
    return prod.mean().item()


def effective_rank(X, n_sample=500):
    Xc = X[:n_sample] - X[:n_sample].mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(Xc)
    p = (S ** 2) / (S ** 2).sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def blur2d(X, sigma):
    if sigma <= 0:
        return X
    N = X.shape[0]
    img = X.view(N, 1, SIDE, SIDE)
    k = max(3, int(6 * sigma) | 1)
    xs = torch.arange(k, dtype=torch.float32, device=X.device) - k // 2
    g1 = torch.exp(-0.5 * (xs / sigma) ** 2)
    g1 = (g1 / g1.sum())
    pad = k // 2
    imgp = F.pad(img, (pad, pad, 0, 0), mode="reflect")
    imgp = F.conv2d(imgp, g1.view(1, 1, 1, k))
    imgp = F.pad(imgp, (0, 0, pad, pad), mode="reflect")
    imgp = F.conv2d(imgp, g1.view(1, 1, k, 1))
    out = imgp.view(N, DIM)
    return out / out.std() * X.std()


def template_separation(seed, sigma):
    """Диагностика ВТОРОГО возможного confound'а (не только rank): размытие
    сглаживает и сами class-шаблоны, потенциально СБЛИЖАЯ их между собой
    (высокочастотное, различающее классы содержимое стирается при большом
    sigma) - если так, задача становится ТРУДНЕЕ независимо от "структуры",
    и падение accuracy(sigma) отражает это, не exploitable-структуру."""
    g = torch.Generator().manual_seed(seed)
    templates = torch.randn(N_CLASSES, DIM, generator=g)
    templates_blurred = blur2d(templates, sigma) if sigma > 0 else templates
    pdist = torch.cdist(templates_blurred, templates_blurred)
    mask = ~torch.eye(N_CLASSES, dtype=torch.bool)
    return pdist[mask].mean().item()


def make_dataset(seed, sigma):
    g = torch.Generator().manual_seed(seed)
    templates = torch.randn(N_CLASSES, DIM, generator=g)
    n_per_class_train = N_TRAIN // N_CLASSES
    n_per_class_test = N_TEST // N_CLASSES
    Xtr, ytr, Xte, yte = [], [], [], []
    for c in range(N_CLASSES):
        Xtr.append(templates[c] + NOISE_STD * torch.randn(n_per_class_train, DIM, generator=g))
        ytr.append(torch.full((n_per_class_train,), c, dtype=torch.long))
        Xte.append(templates[c] + NOISE_STD * torch.randn(n_per_class_test, DIM, generator=g))
        yte.append(torch.full((n_per_class_test,), c, dtype=torch.long))
    Xtr, ytr = torch.cat(Xtr), torch.cat(ytr)
    Xte, yte = torch.cat(Xte), torch.cat(yte)
    perm = torch.randperm(Xtr.shape[0], generator=g)
    Xtr, ytr = Xtr[perm], ytr[perm]

    Xtr = blur2d(Xtr, sigma).to(DEV)
    Xte = blur2d(Xte, sigma).to(DEV)
    ytr, yte = ytr.to(DEV), yte.to(DEV)
    return Xtr, ytr, Xte, yte


class BPNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.l1 = nn.Linear(dims[0], dims[1])
        self.l2 = nn.Linear(dims[1], dims[2])

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


def train_eval_pc(Xtr, ytr, Xte, yte, seed):
    Ytr = torch.zeros(Xtr.shape[0], N_CLASSES, device=DEV)
    Ytr.scatter_(1, ytr.unsqueeze(1), 1.0)
    net = PredictiveCodingNet([DIM, 128, N_CLASSES], seed=seed, adam=True, **PC_KW)
    net.W = [w.to(DEV) for w in net.W]
    net.b = [b.to(DEV) for b in net.b]
    if net.adam:
        net.mW = [m.to(DEV) for m in net.mW]
        net.vW = [v.to(DEV) for v in net.vW]
        net.mb = [m.to(DEV) for m in net.mb]
        net.vb = [v.to(DEV) for v in net.vb]
    for epoch in range(60):
        net.train_step(Xtr, Ytr)
    with torch.no_grad():
        pred = net.forward_pass(Xte).argmax(dim=1)
    return (pred == yte).float().mean().item()


def train_eval_bp(Xtr, ytr, Xte, yte, seed):
    torch.manual_seed(seed)
    net = BPNet([DIM, 128, N_CLASSES]).to(DEV)
    Ytr = torch.zeros(Xtr.shape[0], N_CLASSES, device=DEV)
    Ytr.scatter_(1, ytr.unsqueeze(1), 1.0)
    opt = torch.optim.Adam(net.parameters(), lr=BP_LR)
    for epoch in range(60):
        opt.zero_grad()
        loss = F.mse_loss(net(Xtr), Ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(Xte).argmax(dim=1)
    return (pred == yte).float().mean().item()


def run():
    print("=" * 70)
    print(f"DEV={DEV}  dims=[{DIM},128,{N_CLASSES}]  N_train={N_TRAIN}  N_test={N_TEST}  "
          f"seeds={SEEDS}  epochs=60")
    print("=" * 70)
    rows = []
    for sigma in SIGMAS:
        pc_accs, bp_accs, structs, ranks, tseps = [], [], [], [], []
        t0 = time.time()
        for seed in SEEDS:
            Xtr, ytr, Xte, yte = make_dataset(seed, sigma)
            structs.append(structuredness(Xtr.cpu()))
            ranks.append(effective_rank(Xtr.cpu()))
            tseps.append(template_separation(seed, sigma))
            pc_accs.append(train_eval_pc(Xtr, ytr, Xte, yte, seed))
            bp_accs.append(train_eval_bp(Xtr, ytr, Xte, yte, seed))
        m_struct, m_rank, m_tsep = st.mean(structs), st.mean(ranks), st.mean(tseps)
        m_pc, m_bp = st.mean(pc_accs), st.mean(bp_accs)
        gap = (m_bp - m_pc) * 100
        rows.append((sigma, m_struct, m_rank, m_tsep, m_pc, m_bp, gap))
        print(f"sigma={sigma:4.1f}  structuredness={m_struct:+.4f}  eff_rank={m_rank:6.1f}/784  "
              f"template_sep={m_tsep:6.2f}  "
              f"PC={m_pc*100:5.1f}%  BP={m_bp*100:5.1f}%  gap={gap:+5.1f}п.п.  ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 70)
    print("ИТОГ (sigma, structuredness, eff_rank, template_sep, gap п.п.):")
    for sigma, struct, rank, tsep, pc_acc, bp_acc, gap in rows:
        print(f"  sigma={sigma:4.1f}  structuredness={struct:+.4f}  eff_rank={rank:6.1f}/784  "
              f"template_sep={tsep:6.2f}  gap={gap:+.1f}")
    print("  MNIST (реальный, 9 прогонов): structuredness=+0.5633  gap=+1.79+-0.75")
    print("  M(-1) текст (реальный):       gap=5.9-9.9 (structuredness не сопоставима напрямую - другая модальность)")
    print("=" * 70)


if __name__ == "__main__":
    run()

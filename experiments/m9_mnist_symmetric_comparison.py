"""
M9 prep (docs/ROADMAP.md): перед тем как строить полный 3-группный протокол
M9, нужно закрыть найденную методологическую дыру. `mnist_pc_vs_backprop_sanity.py`
(и его docstring) утверждал, что "разрыв на MNIST заметно меньше 6-10 п.п.
разрыва на синтетике M(-1)" - НО при перепроверке обнаружено ДВЕ проблемы
с этим утверждением:

1. **M(-1) НИКОГДА не был на синтетике.** `data/text/corpus.txt` (реальный
   tinyshakespeare, 1115394 байт) используется как источник данных ВО ВСЕХ
   файлах M(-1) (`m_minus1_scaling_study.py`, `_larger.py`,
   `_symmetric_comparison.py`) с самого первого прогона - "6-10 п.п. на
   синтетике" было фактически неверной характеристикой в докстрингах,
   унаследованной без проверки. Разрыв M(-1) - это разрыв на РЕАЛЬНОМ
   тексте (char-LM), не на синтетических случайных векторах.

2. **BP-часть mnist_pc_vs_backprop_sanity.py использовала ОДИН фиксированный
   lr=0.001, без sweep** - ровно та же асимметричная строгость в пользу PC,
   что уже была найдена и исправлена для M(-1) (см. "Найдена более
   серьёзная проблема методологии" в VERIFICATION_LOG - там неоткалиброванный
   BP dал ложное "сужение разрыва", отозвано после честного lr-sweep).
   Никогда не применялась та же поправка к MNIST-тесту.

Здесь: тот же протокол, что и в `m_minus1_symmetric_comparison.py`
(фаза 1 - lr-sweep для BP на ПОЛНОМ бюджете, фаза 2 - честное сравнение
по эпохам с лучшим lr), применённый к MNIST-тесту, прежде чем делать
какие-либо выводы о "разрыв меньше на реальных данных".
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from core.predictive_coding import PredictiveCodingNet
from core.mnist_loader import load_mnist

torch.manual_seed(42)

N_TRAIN = 2000
N_TEST = 1000
DIMS = [784, 128, 10]
EPOCHS = 150


class BPNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dims[l], dims[l + 1]) for l in range(len(dims) - 1))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


def accuracy(logits, y_idx):
    return (logits.argmax(dim=1) == y_idx).float().mean().item()


def load_data():
    tr_x, tr_y, te_x, te_y = load_mnist()
    g = torch.Generator().manual_seed(1)
    train_idx = torch.randperm(tr_x.shape[0], generator=g)[:N_TRAIN]
    test_idx = torch.randperm(te_x.shape[0], generator=g)[:N_TEST]

    Xtr = tr_x[train_idx].reshape(N_TRAIN, -1)
    ytr_idx = tr_y[train_idx]
    Ytr = torch.zeros(N_TRAIN, 10)
    Ytr.scatter_(1, ytr_idx.unsqueeze(1), 1.0)

    Xte = te_x[test_idx].reshape(N_TEST, -1)
    yte_idx = te_y[test_idx]
    Yte = torch.zeros(N_TEST, 10)
    Yte.scatter_(1, yte_idx.unsqueeze(1), 1.0)
    return Xtr, Ytr, ytr_idx, Xte, Yte, yte_idx


def train_bp(Xtr, Ytr, yte_idx, Xte, lr, epochs=EPOCHS):
    bp = BPNet(DIMS)
    opt = torch.optim.Adam(bp.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(bp(Xtr), Ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = accuracy(bp(Xte), yte_idx)
    return bp, acc


def run():
    print("Загрузка реального MNIST...")
    Xtr, Ytr, ytr_idx, Xte, Yte, yte_idx = load_data()
    print(f"N_train={N_TRAIN}  N_test={N_TEST}  dims={DIMS}")

    print("=" * 70)
    print(f"ФАЗА 1: lr-sweep для BP (budget={EPOCHS} эпох) - та же строгость, что PC")
    print("=" * 70)
    bp_lr_results = []
    for lr in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]:
        t0 = time.time()
        _, acc = train_bp(Xtr, Ytr, yte_idx, Xte, lr)
        print(f"  BP lr={lr:.4f}  acc={acc*100:5.1f}%  ({time.time()-t0:.1f}s)")
        bp_lr_results.append((lr, acc))
    best_bp_lr = max(bp_lr_results, key=lambda r: r[1])[0]
    print(f"  -> лучший BP lr: {best_bp_lr} (использовавшийся ранее без sweep: 0.001)")

    print("\n" + "=" * 70)
    print(f"ФАЗА 2: симметричное сравнение по эпохам (PC vs BP lr={best_bp_lr})")
    print("=" * 70)
    pc = PredictiveCodingNet(DIMS, relax_steps=20, relax_lr=0.08, weight_lr=0.01, seed=1, adam=True, weight_decay=0.02)
    bp = BPNet(DIMS)
    opt = torch.optim.Adam(bp.parameters(), lr=best_bp_lr)

    CHECKPOINTS = [10, 30, 50, 100, 150]
    ep = 0
    for target_ep in CHECKPOINTS:
        while ep < target_ep:
            pc.train_step(Xtr, Ytr)
            opt.zero_grad()
            loss = nn.functional.mse_loss(bp(Xtr), Ytr)
            loss.backward()
            opt.step()
            ep += 1
        pc_acc = accuracy(pc.forward_pass(Xte), yte_idx)
        with torch.no_grad():
            bp_acc = accuracy(bp(Xte), yte_idx)
        print(f"epoch={ep:4d}  PC={pc_acc*100:5.1f}%  BP(lr={best_bp_lr})={bp_acc*100:5.1f}%  "
              f"разрыв={(bp_acc-pc_acc)*100:+.1f} п.п.")

    print("=" * 70)
    print("Для сравнения: старый (неоткалиброванный BP lr=0.001, без sweep) разрыв "
          "документирован в VERIFICATION_LOG как 'заметно меньше 6-10 п.п.' - "
          "смотрим, выживает ли это утверждение при честном BP.")
    print("=" * 70)


if __name__ == "__main__":
    run()

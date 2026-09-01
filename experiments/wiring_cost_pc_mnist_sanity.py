"""
Продолжение дендритной компартментализации (dendritic_pc_mnist_sanity.py) -
веб-поиск 2026-09-01 нашёл смежную идею из коннектомики: "wiring cost" -
реальные связи мозга штрафуются пропорционально ФИЗИЧЕСКОМУ расстоянию, а
не жёстко режутся по границам патчей. Дальние связи РЕДКИ, но не запрещены
(в отличие от жёсткой w0_mask, которая физически обнуляет связи вне патча
навсегда).

Реализовано с нуля: `PredictiveCodingNet(wiring_cost=..., wiring_lambda=...)` -
мягкий L1-штраф на веса первого слоя, пропорциональный евклидову расстоянию
между позицией скрытого нейрона (случайно назначенная "координата" на
холсте 28x28) и позицией входного пикселя.

Вопрос: даёт ли МЯГКИЙ, обучаемый штраф за расстояние лучший компромисс
точность/локальность, чем ЖЁСТКАЯ маска патчей при том же ЭФФЕКТИВНОМ уровне
разреженности (доля весов с |W| выше порога)?
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.predictive_coding import PredictiveCodingNet
from core.mnist_loader import load_mnist

torch.manual_seed(42)

N_TRAIN = 2000
N_TEST = 1000
HIDDEN = 128
IMG_SIZE = 28
EPOCHS = 150


def accuracy(logits, y_idx):
    return (logits.argmax(dim=1) == y_idx).float().mean().item()


def build_wiring_cost(hidden_dim, img_size=IMG_SIZE, seed=1):
    g = torch.Generator().manual_seed(seed)
    hidden_pos = torch.rand(hidden_dim, 2, generator=g) * img_size
    ys, xs = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing="ij")
    input_pos = torch.stack([ys.flatten(), xs.flatten()], dim=1).float()
    dist = torch.cdist(hidden_pos, input_pos)  # (hidden_dim, img_size*img_size)
    return dist / dist.max()  # нормируем в [0,1]


def run():
    tr_x, tr_y, te_x, te_y = load_mnist()
    g = torch.Generator().manual_seed(1)
    train_idx = torch.randperm(tr_x.shape[0], generator=g)[:N_TRAIN]
    test_idx = torch.randperm(te_x.shape[0], generator=g)[:N_TEST]
    Xtr = tr_x[train_idx].reshape(N_TRAIN, -1)
    ytr_idx = tr_y[train_idx]
    Ytr = torch.zeros(N_TRAIN, 10); Ytr.scatter_(1, ytr_idx.unsqueeze(1), 1.0)
    Xte = te_x[test_idx].reshape(N_TEST, -1)
    yte_idx = te_y[test_idx]

    dims = [784, HIDDEN, 10]
    wiring_cost = build_wiring_cost(HIDDEN)

    print("Полносвязный baseline (уже известно): 88.4%")
    print("Жёсткая дендритная маска (уже известно): 2x2=80.5%, 4x4=75.0%, 7x7=75.4%")
    print()
    print("Мягкий wiring-cost штраф (wiring_lambda sweep)...")
    for wl in [0.0, 0.0005, 0.001, 0.003, 0.01, 0.03]:
        pc = PredictiveCodingNet(dims, relax_steps=20, relax_lr=0.08, weight_lr=0.01, seed=1,
                                  adam=True, weight_decay=0.02, wiring_cost=wiring_cost, wiring_lambda=wl)
        for ep in range(EPOCHS):
            pc.train_step(Xtr, Ytr)
        acc = accuracy(pc.forward_pass(Xte), yte_idx)
        # эффективная разреженность - доля весов, чья |W| упала ниже 1% от макс |W|
        w0 = pc.W[0].abs()
        sparsity = (w0 < 0.01 * w0.max()).float().mean().item()
        print(f"  wiring_lambda={wl:.4f}: test_acc={acc*100:5.1f}%  "
              f"эффективно-разреженных весов (|W|<1%max)={sparsity*100:5.1f}%")


if __name__ == "__main__":
    run()

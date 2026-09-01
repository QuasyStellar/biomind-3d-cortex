"""
Дендритная компартментализация (веб-поиск 2026-09-01: "Dendritic Localized
Learning", ICML 2025 - локальные нелинейные дендритные ветви вместо
полносвязного слоя, ближе к реальной биологии одиночного нейрона) - НОВЫЙ
механизм, реализован с нуля в core/predictive_coding.py (`w0_mask`), никогда
раньше не пробовался в проекте (rule 2: искать в литературе то, что могли
упустить).

Первый слой генома PredictiveCodingNet СДЕЛАН block-структурным: каждая
группа скрытых нейронов ("дендритная ветвь") видит только СВОЙ
пространственный патч изображения (7x7 из 28x28, а не весь кадр) - вместо
полносвязного 784->128. В ~16 раз меньше эффективных параметров в первом
слое (6272 против 100352).

Вопрос: насколько теряет точность дендритная (компартментализованная)
структура по сравнению с полносвязной на РЕАЛЬНОМ MNIST (тот же протокол,
что и mnist_pc_vs_backprop_sanity.py, N_train=2000/N_test=1000) - и стоит
ли это компромисса, если разница мала при значительно меньшем числе связей?
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
PATCH_GRID = 4  # 4x4=16 патчей по 7x7
EPOCHS = 150


def accuracy(logits, y_idx):
    return (logits.argmax(dim=1) == y_idx).float().mean().item()


def spatial_patch_mask(hidden_dim, img_size=IMG_SIZE, patch_grid=PATCH_GRID):
    patch_size = img_size // patch_grid
    n_patches = patch_grid * patch_grid
    units_per_patch = hidden_dim // n_patches
    mask = torch.zeros(hidden_dim, img_size * img_size)
    for p in range(n_patches):
        py, px = p // patch_grid, p % patch_grid
        idx = []
        for dy in range(patch_size):
            for dx in range(patch_size):
                y, x = py * patch_size + dy, px * patch_size + dx
                idx.append(y * img_size + x)
        idx = torch.tensor(idx)
        h0, h1 = p * units_per_patch, (p + 1) * units_per_patch
        mask[h0:h1][:, idx] = 1.0
    return mask


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
    mask = spatial_patch_mask(HIDDEN)
    n_params_full = 784 * HIDDEN
    n_params_dendritic = int(mask.sum().item())
    print(f"Параметров в первом слое: полносвязный={n_params_full}  дендритный={n_params_dendritic} "
          f"({100*n_params_dendritic/n_params_full:.1f}%)")

    print("\n1) Полносвязный геном (baseline, уже установленный лучший конфиг)...")
    pc_full = PredictiveCodingNet(dims, relax_steps=20, relax_lr=0.08, weight_lr=0.01, seed=1, adam=True, weight_decay=0.02)
    for ep in range(EPOCHS):
        pc_full.train_step(Xtr, Ytr)
    acc_full = accuracy(pc_full.forward_pass(Xte), yte_idx)
    print(f"   test_acc={acc_full*100:.1f}%")

    print("2) Дендритный геном (первый слой - block-структурный, 16 пространственных патчей)...")
    pc_dend = PredictiveCodingNet(dims, relax_steps=20, relax_lr=0.08, weight_lr=0.01, seed=1, adam=True, weight_decay=0.02, w0_mask=mask)
    for ep in range(EPOCHS):
        pc_dend.train_step(Xtr, Ytr)
    acc_dend = accuracy(pc_dend.forward_pass(Xte), yte_idx)
    print(f"   test_acc={acc_dend*100:.1f}%")

    print("=" * 70)
    print(f"Полносвязный: {acc_full*100:.1f}%  ({n_params_full} параметров в 1 слое)")
    print(f"Дендритный:   {acc_dend*100:.1f}%  ({n_params_dendritic} параметров в 1 слое, {100*n_params_dendritic/n_params_full:.1f}%)")
    print(f"Разница: {(acc_full-acc_dend)*100:+.1f} п.п. за {100*(1-n_params_dendritic/n_params_full):.0f}% сокращение параметров")
    print("=" * 70)

    return acc_full, acc_dend


if __name__ == "__main__":
    run()

"""
Самая первая проверка с нуля: работает ли Hebbian-предиктивная релаксация
(Whittington & Bogacz) без единого .backward() сопоставимо с обычным
backprop на одной и той же архитектуре и той же задаче?

Задача: 2D спираль, 3 класса — классическая нелинейная многоклассовая
задача, где однослойная сеть не работает, а credit assignment через
глубину (2 скрытых слоя) реально нужен. Обе сети: 2 -> 32 -> 32 -> 3,
tanh на скрытых, линейный выход, MSE к one-hot цели (одна и та же
функция потерь для честного сравнения).

Если PC не достигает сопоставимой точности при том же бюджете (число
эпох) — весь ARCHITECTURE.md, построенный на "backprop не нужен нигде",
нужно пересматривать. Это и есть цель этого скрипта.
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from core.predictive_coding import PredictiveCodingNet

torch.manual_seed(42)


def make_spiral(n_per_class=150, n_classes=3, noise=0.2, seed=42):
    g = torch.Generator().manual_seed(seed)
    X = torch.zeros(n_per_class * n_classes, 2)
    y_idx = torch.zeros(n_per_class * n_classes, dtype=torch.long)
    for c in range(n_classes):
        r = torch.linspace(0.05, 1.0, n_per_class)
        theta = torch.linspace(c * 4, (c + 1) * 4, n_per_class) + torch.randn(n_per_class, generator=g) * noise
        ix = slice(c * n_per_class, (c + 1) * n_per_class)
        X[ix, 0] = r * torch.sin(theta)
        X[ix, 1] = r * torch.cos(theta)
        y_idx[ix] = c
    onehot = torch.zeros(n_per_class * n_classes, n_classes)
    onehot.scatter_(1, y_idx.unsqueeze(1), 1.0)
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], onehot[perm], y_idx[perm]


class BPNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for l in range(len(dims) - 1):
            layers.append(nn.Linear(dims[l], dims[l + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


def accuracy(logits, y_idx):
    return (logits.argmax(dim=1) == y_idx).float().mean().item()


def run():
    X, Y_onehot, y_idx = make_spiral(n_per_class=150, n_classes=3)
    n_train = int(0.8 * X.shape[0])
    Xtr, Ytr, ytr_idx = X[:n_train], Y_onehot[:n_train], y_idx[:n_train]
    Xte, Yte, yte_idx = X[n_train:], Y_onehot[n_train:], y_idx[n_train:]

    dims = [2, 32, 32, 3]
    EPOCHS = 300

    # --- Predictive Coding: zero backward(), full-batch update per epoch, Adam on local grad ---
    pc = PredictiveCodingNet(dims, relax_steps=50, relax_lr=0.1, weight_lr=0.01, seed=1, adam=True)
    pc_train_acc, pc_test_acc, pc_energy = [], [], []
    t0 = time.time()
    for ep in range(EPOCHS):
        e = pc.train_step(Xtr, Ytr)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            pc_train_acc.append((ep, accuracy(pc.forward_pass(Xtr), ytr_idx)))
            pc_test_acc.append((ep, accuracy(pc.forward_pass(Xte), yte_idx)))
            pc_energy.append((ep, e))
    pc_time = time.time() - t0
    print(f"PC relaxation energy: start={pc_energy[0][1]:.4f} -> end={pc_energy[-1][1]:.4f}")

    # --- Backprop baseline: identical architecture, identical loss (MSE) ---
    bp = BPNet(dims)
    opt = torch.optim.Adam(bp.parameters(), lr=0.01)
    bp_train_acc, bp_test_acc = [], []
    t0 = time.time()
    for ep in range(EPOCHS):
        opt.zero_grad()
        out = bp(Xtr)
        loss = nn.functional.mse_loss(out, Ytr)
        loss.backward()
        opt.step()
        if ep % 5 == 0 or ep == EPOCHS - 1:
            with torch.no_grad():
                bp_train_acc.append((ep, accuracy(bp(Xtr), ytr_idx)))
                bp_test_acc.append((ep, accuracy(bp(Xte), yte_idx)))
    bp_time = time.time() - t0

    pc_final = pc_test_acc[-1][1]
    bp_final = bp_test_acc[-1][1]

    print("=" * 70)
    print(f"PC  (zero backward, {EPOCHS} epochs, {pc_time:.1f}s): "
          f"train={pc_train_acc[-1][1]*100:.1f}%  test={pc_final*100:.1f}%")
    print(f"BP  (standard backprop, {EPOCHS} epochs, {bp_time:.1f}s): "
          f"train={bp_train_acc[-1][1]*100:.1f}%  test={bp_final*100:.1f}%")
    print(f"Gap (BP - PC) on test accuracy: {(bp_final - pc_final)*100:+.1f} p.p.")
    print("=" * 70)

    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots"), exist_ok=True)
    plot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots", "pc_vs_backprop_sanity.png")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(*zip(*pc_test_acc), label=f"PC test acc (final {pc_final*100:.1f}%)", color="darkorange", linewidth=2)
    ax.plot(*zip(*bp_test_acc), label=f"BP test acc (final {bp_final*100:.1f}%)", color="steelblue", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Predictive Coding (zero backward()) vs Backprop — 3-class spiral, matched architecture/loss/epochs")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")

    return pc_final, bp_final, EPOCHS, n_train, Xte.shape[0]


if __name__ == "__main__":
    run()

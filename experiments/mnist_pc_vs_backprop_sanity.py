"""
Первое подключение РЕАЛЬНЫХ сенсорных данных в проекте (README, пункт
"подключить реальные сенсорные данные хотя бы MNIST-уровня") - до сих пор
ВСЕ тесты (M(-1) включительно) шли на синтетических данных (случайные
вектора, спирали). Вопрос: держится ли PC-vs-BP разрыв (см. M(-1) в
VERIFICATION_LOG - 6-10 п.п., "примерно постоянный") на РЕАЛЬНЫХ пикселях
с их реальной статистикой (корреляции между соседними пикселями, реальные
классы), а не на синтетике?

Данные - настоящий MNIST (`core/mnist_loader.py`, сырой IDX-формат, скачан
напрямую, без torchvision). Toy-масштаб (N_train=2000, N_test=1000, явно
указано - не "решённый MNIST", а честная проверка на реальных данных
скромного объёма, тот же принцип "явный N", что и во всех остальных тестах.

Обе сети: 784 -> 128 -> 10, tanh на скрытом, линейный выход, MSE к one-hot -
идентичная архитектура и функция потерь для PC и backprop (тот же протокол,
что и в pc_vs_backprop_sanity.py, только данные другие).
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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


def run():
    print("Загрузка реального MNIST...")
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
    print(f"N_train={N_TRAIN}  N_test={N_TEST}  dims={DIMS}")

    # --- PC: zero backward(), full-batch, тот же протокол, что в M(-1) ---
    pc = PredictiveCodingNet(DIMS, relax_steps=20, relax_lr=0.08, weight_lr=0.01, seed=1, adam=True, weight_decay=0.02)
    pc_test_acc = []
    t0 = time.time()
    for ep in range(EPOCHS):
        pc.train_step(Xtr, Ytr)
        if ep % 10 == 0 or ep == EPOCHS - 1:
            acc = accuracy(pc.forward_pass(Xte), yte_idx)
            pc_test_acc.append((ep, acc))
            print(f"  PC  ep={ep:3d}  test_acc={acc*100:5.1f}%  ({time.time()-t0:.1f}s)")
    pc_time = time.time() - t0

    # --- BP: тот же протокол ---
    bp = BPNet(DIMS)
    opt = torch.optim.Adam(bp.parameters(), lr=0.001)
    bp_test_acc = []
    t0 = time.time()
    for ep in range(EPOCHS):
        opt.zero_grad()
        out = bp(Xtr)
        loss = nn.functional.mse_loss(out, Ytr)
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            with torch.no_grad():
                acc = accuracy(bp(Xte), yte_idx)
            bp_test_acc.append((ep, acc))
            print(f"  BP  ep={ep:3d}  test_acc={acc*100:5.1f}%  ({time.time()-t0:.1f}s)")
    bp_time = time.time() - t0

    pc_final, bp_final = pc_test_acc[-1][1], bp_test_acc[-1][1]
    print("=" * 70)
    print(f"PC (zero backward, {EPOCHS} эпох, {pc_time:.1f}s): test={pc_final*100:.1f}%")
    print(f"BP (backprop, {EPOCHS} эпох, {bp_time:.1f}s): test={bp_final*100:.1f}%")
    print(f"Разрыв (BP - PC): {(bp_final-pc_final)*100:+.1f} п.п.")
    print(f"(для сравнения: разрыв на синтетике в M(-1) держался 6-10 п.п. на всех бюджетах)")
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(*zip(*pc_test_acc), label=f"PC (final {pc_final*100:.1f}%)", color="darkorange", linewidth=2)
    ax.plot(*zip(*bp_test_acc), label=f"BP (final {bp_final*100:.1f}%)", color="steelblue", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (real MNIST)")
    ax.set_title(f"PC vs BP на РЕАЛЬНОМ MNIST (N_train={N_TRAIN}, N_test={N_TEST})")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(plots_dir, "mnist_pc_vs_backprop_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")

    return pc_final, bp_final


if __name__ == "__main__":
    run()

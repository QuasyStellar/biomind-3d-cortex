"""
Synaptic Intelligence (Zenke et al. 2017, найдено веб-поиском 2026-09-01,
реализовано с нуля в core/predictive_coding.py) - НИКОГДА раньше не
пробовалось в проекте. Прямой тест ключевого обещания проекта (README:
"должен обучать динамически... запоминать", "point-editing" в 8-осевой
таблице ROADMAP): последовательное обучение на ДВУХ задачах подряд (не
i.i.d. вперемешку) - классический сетап для катастрофического забывания.

Задача A: 2-классовая спираль (вариант 1). Задача B: другая 2-классовая
спираль (вариант 2, ДРУГОЙ seed/геометрия). Протокол:
  1. Обучить на A (N шагов) -> измерить acc(A).
  2. Обучить на B (N шагов) - A si_new_task() перед этим для SI-версии -
     измерить acc(B) сразу после.
  3. Измерить acc(A) СНОВА (забыла ли сеть A, пока училась B?).
Сравниваем: baseline (без SI) vs SI-enabled, одна и та же архитектура/
данные/бюджет для обоих.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.predictive_coding import PredictiveCodingNet

torch.manual_seed(42)


def make_spiral(n_per_class=100, n_classes=2, noise=0.2, seed=42, rotation=0.0):
    g = torch.Generator().manual_seed(seed)
    X = torch.zeros(n_per_class * n_classes, 2)
    y_idx = torch.zeros(n_per_class * n_classes, dtype=torch.long)
    for c in range(n_classes):
        r = torch.linspace(0.05, 1.0, n_per_class)
        theta = torch.linspace(c * 4, (c + 1) * 4, n_per_class) + rotation + torch.randn(n_per_class, generator=g) * noise
        ix = slice(c * n_per_class, (c + 1) * n_per_class)
        X[ix, 0] = r * torch.sin(theta)
        X[ix, 1] = r * torch.cos(theta)
        y_idx[ix] = c
    onehot = torch.zeros(n_per_class * n_classes, n_classes)
    onehot.scatter_(1, y_idx.unsqueeze(1), 1.0)
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], onehot[perm], y_idx[perm]


def accuracy(net, X, y_idx):
    logits = net.forward_pass(X)
    return (logits.argmax(dim=1) == y_idx).float().mean().item()


def run(si_lambda=1.0, epochs=200, seed=1):
    Xa, Ya, ya_idx = make_spiral(seed=1, rotation=0.0)
    Xb, Yb, yb_idx = make_spiral(seed=2, rotation=3.0)  # другая геометрия - другая "задача"

    dims = [2, 32, 2]

    def train_sequential(si_enabled):
        net = PredictiveCodingNet(dims, relax_steps=40, relax_lr=0.1, weight_lr=0.01, seed=seed,
                                   adam=True, weight_decay=0.01, si_enabled=si_enabled, si_lambda=si_lambda)
        for ep in range(epochs):
            net.train_step(Xa, Ya)
        acc_a_after_a = accuracy(net, Xa, ya_idx)

        if si_enabled:
            net.si_new_task()

        for ep in range(epochs):
            net.train_step(Xb, Yb)
        acc_b_after_b = accuracy(net, Xb, yb_idx)
        acc_a_after_b = accuracy(net, Xa, ya_idx)  # забыла ли A?

        return acc_a_after_a, acc_b_after_b, acc_a_after_b

    print("=" * 70)
    print("Baseline (без Synaptic Intelligence)...")
    a1_base, b_base, a2_base = train_sequential(si_enabled=False)
    print(f"  acc(A) после обучения A:        {a1_base*100:5.1f}%")
    print(f"  acc(B) после обучения B:        {b_base*100:5.1f}%")
    print(f"  acc(A) ПОСЛЕ обучения B (забыла?): {a2_base*100:5.1f}%")
    forgetting_base = a1_base - a2_base
    print(f"  Забывание (acc(A) до - после B): {forgetting_base*100:+.1f} п.п.")

    print("\nSynaptic Intelligence (si_lambda={})...".format(si_lambda))
    a1_si, b_si, a2_si = train_sequential(si_enabled=True)
    print(f"  acc(A) после обучения A:        {a1_si*100:5.1f}%")
    print(f"  acc(B) после обучения B:        {b_si*100:5.1f}%")
    print(f"  acc(A) ПОСЛЕ обучения B (забыла?): {a2_si*100:5.1f}%")
    forgetting_si = a1_si - a2_si
    print(f"  Забывание (acc(A) до - после B): {forgetting_si*100:+.1f} п.п.")

    print("=" * 70)
    print(f"Забывание: baseline={forgetting_base*100:+.1f}п.п.  SI={forgetting_si*100:+.1f}п.п.")
    print(f"Точность на B: baseline={b_base*100:.1f}%  SI={b_si*100:.1f}% (SI не должен сильно мешать B)")
    if forgetting_si < forgetting_base - 0.03:
        print("=> SI РЕАЛЬНО снижает забывание A по сравнению с baseline")
    elif forgetting_si > forgetting_base + 0.03:
        print("=> SI УХУДШАЕТ ситуацию (забывание больше, чем без SI) - неожиданно, зафиксировано честно")
    else:
        print("=> Разница незначительна на этом прогоне")
    print("=" * 70)
    return forgetting_base, forgetting_si, b_base, b_si


if __name__ == "__main__":
    run()

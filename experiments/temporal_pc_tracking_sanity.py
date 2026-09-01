"""
Первая честная проверка `core/temporal_pc.py` (Millidge et al. 2024,
статья прочитана напрямую - см. docstring temporal_pc.py) - не на реальной
задаче ещё, а на том же ТИПЕ игрушечной задачи, на которой валидировалась
сама статья: линейное слежение за скрытым состоянием через "спутанные"
(случайно перемешанные) наблюдения.

Истинная динамика: 2D-вращение (x_{t+1} = R_theta @ x_t) - простая,
негладкая по компонентам, но полностью предсказуемая последовательность.
Наблюдение: y_t = C_true @ x_t (проекция в 5-мерное пространство ФИКСИРОВАННОЙ
случайной матрицей + шум) - сеть НИКОГДА не видит x_t напрямую, только y_t,
и не знает ни A_true, ни C_true - должна выучить их местные аналоги (A, C)
ЧЕРЕЗ ЧИСТО ЛОКАЛЬНОЕ (Hebbian) правило, без единого .backward() по времени.

Честная проверка: снижается ли ошибка one-step-ahead предсказания y_t по
ходу потока (обучение online, один проход, не эпохи по одному и тому же
куску) - и превосходит ли обучаемая версия версию с ЗАМОРОЖЕННЫМИ
случайными A/C (baseline "без обучения вообще", изолирует эффект именно
локального правила, не архитектуры/релаксации самой по себе).
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.temporal_pc import TemporalPredictiveCoding


def make_sequence(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    theta = 0.15
    R = torch.tensor([[torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
                       [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))]])
    C_true = torch.randn(5, 2, generator=g) * 0.8
    x = torch.tensor([1.0, 0.0])
    xs, ys = [], []
    for t in range(T):
        x = R @ x
        y = C_true @ x + 0.05 * torch.randn(5, generator=g)
        xs.append(x.clone())
        ys.append(y.clone())
    return xs, ys


def run(T=800, seed=1):
    xs, ys = make_sequence(T, seed=seed)

    net_learn = TemporalPredictiveCoding(x_dim=2, y_dim=5, relax_steps=10, relax_dt=0.1,
                                          weight_lr=0.02, seed=seed, nonlinear=False)
    net_frozen = TemporalPredictiveCoding(x_dim=2, y_dim=5, relax_steps=10, relax_dt=0.1,
                                           weight_lr=0.0, seed=seed, nonlinear=False)

    pred_errs_learn, pred_errs_frozen = [], []
    t0 = time.time()
    for t in range(T):
        y_t = ys[t]
        # one-step-ahead: предсказание ДО того, как сеть увидит y_t в этом шаге
        pred_learn = net_learn.predict_y()
        pred_frozen = net_frozen.predict_y()
        pred_errs_learn.append((pred_learn - y_t).pow(2).sum().item())
        pred_errs_frozen.append((pred_frozen - y_t).pow(2).sum().item())

        net_learn.step(y_t)
        net_frozen.step(y_t)

    print(f"T={T} шагов, {time.time()-t0:.1f}s")
    early_learn = sum(pred_errs_learn[:100]) / 100
    late_learn = sum(pred_errs_learn[-100:]) / 100
    early_frozen = sum(pred_errs_frozen[:100]) / 100
    late_frozen = sum(pred_errs_frozen[-100:]) / 100
    print(f"learn : early(0-100)={early_learn:.4f}  late(700-800)={late_learn:.4f}  "
          f"({'СНИЖАЕТСЯ' if late_learn < early_learn * 0.7 else 'не снижается явно'})")
    print(f"frozen: early(0-100)={early_frozen:.4f}  late(700-800)={late_frozen:.4f}")
    print(f"Итог: late_learn/late_frozen = {late_learn/late_frozen:.3f} "
          f"({'обучение явно лучше замороженного' if late_learn < late_frozen * 0.7 else 'разница не решительная'})")
    return early_learn, late_learn, early_frozen, late_frozen


if __name__ == "__main__":
    run()

"""
Temporal Predictive Coding (tPC) - Millidge, Tang, Osanlouy, Harper, Bogacz,
"Predictive coding networks for temporal prediction", PLoS Comput Biol 2024
(doi.org/10.1371/journal.pcbi.1011183) - реализовано с нуля СТРОГО по
уравнениям статьи (прочитана напрямую, не по пересказу в
CLAUDE_RESEARCH_SPEC.md - тот пересказ терял precision-weighting и путал
временную/пространственную структуру, честно зафиксировано в
docs/VERIFICATION_LOG.md).

Модель пространства состояний, не просто "слой предсказывает слой":
скрытое состояние x_k эволюционирует во времени через матрицу перехода A
(+ опционально control input u_k через B), НАБЛЮДАЕМЫЙ сигнал y_k
получается из x_k через отдельную матрицу C - ровно та же структура, что
в фильтре Калмана, но БЕЗ распространения полной гауссовой апостериорной
ковариации (Dirac-delta prior на x_{k-1} вместо гауссова посредника) -
статья явно называет это MAP-упрощением, не идентичным KF.

Уравнения (статья, eq. 8-11):
  eps_y = (y_k - C @ f(x_k)) * Sigma_y_inv
  eps_x = (x_k - A @ f(x_hat_{k-1}) - B @ u_k) * Sigma_x_inv
  dx_k/dt = -x_k + A^T @ (eps_x * Sigma_x_inv) + C^T @ (eps_y * Sigma_y_inv)
(релаксация x_k градиентным потоком к MAP-оценке ПРЕЖДЕ обновления весов)
  dA = eta * outer(eps_x * Sigma_x_inv, f(x_hat_{k-1}))
  dC = eta * outer(eps_y * Sigma_y_inv, x_hat_k)
  dB = eta * outer(eps_x * Sigma_x_inv, u_k)
Обучение - ПОСЛЕ схождения релаксации на каждом шаге k, не непрерывно
(статья: "purely Hebbian plasticity", только локальная пре/пост-
синаптическая информация) - ни одного .backward() по времени (никакого
BPTT), в отличие от обычных RNN/LSTM.
"""
import torch


def act(z):
    return torch.tanh(z)


def act_deriv(z):
    t = torch.tanh(z)
    return 1.0 - t * t


class TemporalPredictiveCoding:
    def __init__(self, x_dim, y_dim, u_dim=0, relax_steps=10, relax_dt=0.1,
                 weight_lr=0.01, seed=0, nonlinear=True, control=False):
        """x_dim: размерность скрытого состояния (латент). y_dim: размерность
        наблюдения. u_dim: размерность control input (0, если нет - control
        применяется только при control=True). nonlinear: f()=tanh (статья
        тестирует ОБА варианта - линейный и нелинейный, нелинейный лучше
        на "large-amplitude oscillations" - нелинейный маятник). Sigma_x/
        Sigma_y - precision (обратная дисперсия) ошибок - здесь начинаем с
        identity (упрощение v1, статья использует их для взвешивания по
        неопределённости - честно отмечено, не implементировано в этой
        версии, открытый следующий шаг)."""
        g = torch.Generator().manual_seed(seed)
        self.x_dim, self.y_dim, self.u_dim = x_dim, y_dim, u_dim
        self.relax_steps = relax_steps
        self.relax_dt = relax_dt
        self.weight_lr = weight_lr
        self.nonlinear = nonlinear
        self.control = control
        self.A = torch.randn(x_dim, x_dim, generator=g) * (1.0 / x_dim ** 0.5)
        self.C = torch.randn(y_dim, x_dim, generator=g) * (1.0 / x_dim ** 0.5)
        self.B = torch.randn(x_dim, max(u_dim, 1), generator=g) * (1.0 / x_dim ** 0.5) if control else None
        self.x_hat_prev = torch.zeros(x_dim)  # x_hat_{k-1}, оценка предыдущего шага (Dirac-delta prior)

    def _f(self, x):
        return act(x) if self.nonlinear else x

    def _f_deriv(self, x):
        return act_deriv(x) if self.nonlinear else torch.ones_like(x)

    @torch.no_grad()
    def step(self, y_k, u_k=None):
        """Один временной шаг k: релаксация x_k к MAP-оценке (эту точку статья
        называет x_hat_k), затем ЛОКАЛЬНОЕ (Hebbian) обновление A/C/(B) - ни
        одного .backward(), ни по слоям, ни по времени."""
        x_prev_f = self._f(self.x_hat_prev)
        u = u_k if (self.control and u_k is not None) else torch.zeros(max(self.u_dim, 1))

        x = self.A @ x_prev_f  # feedforward-инициализация релаксации (как и в spatial PC)
        if self.control:
            x = x + self.B @ u

        for _ in range(self.relax_steps):
            fx = self._f(x)
            eps_y = y_k - self.C @ fx  # sensory error (eq. 8), Sigma_y=I упрощение
            eps_x = x - self.A @ x_prev_f - (self.B @ u if self.control else 0.0)  # temporal error (eq. 9)
            dx = -x + self.A.T @ eps_x + (self.C.T @ eps_y) * self._f_deriv(x)  # eq. 10, gradient flow
            x = x + self.relax_dt * dx

        x_hat_k = x
        fx_final = self._f(x_hat_k)
        eps_y = y_k - self.C @ fx_final
        eps_x = x_hat_k - self.A @ x_prev_f - (self.B @ u if self.control else 0.0)

        # Локальные Hebbian-обновления ПОСЛЕ схождения (eq. 11) - Sigma=I упрощение
        self.A += self.weight_lr * torch.outer(eps_x, x_prev_f)
        self.C += self.weight_lr * torch.outer(eps_y, x_hat_k)
        if self.control:
            self.B += self.weight_lr * torch.outer(eps_x, u)

        self.x_hat_prev = x_hat_k.detach().clone()
        energy = (eps_x.pow(2).sum() + eps_y.pow(2).sum()).item()
        return x_hat_k, energy

    def predict_y(self, x=None):
        """Наблюдение, предсказанное текущим (или заданным) скрытым состоянием."""
        x = self.x_hat_prev if x is None else x
        return self.C @ self._f(x)

    def reset_state(self):
        self.x_hat_prev = torch.zeros(self.x_dim)

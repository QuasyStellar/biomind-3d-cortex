"""
Supervised Predictive Coding (Whittington & Bogacz, 2017) — с нуля,
без единого вызова .backward(). Локальное правило: вес обновляется
по внешнему произведению (локальная ошибка * производная активации)
и пресинаптической активности — вычисляется вручную, не через autograd.

Слои x_0 (вход, зажат) ... x_L (цель, зажата во время обучения).
Предсказание идёт снизу вверх: x_hat_l = f(W_l @ x_{l-1} + b_l).
Ошибка: eps_l = x_l - x_hat_l. Энергия: F = 0.5 * sum_l ||eps_l||^2.
Свободные слои (1..L-1) релаксируют градиентным спуском по F
относительно x_l (это НЕ обучение весов, это вывод/inference состояния).
После релаксации веса обновляются локальным правилом = -dF/dW_l.
"""
import torch


def act(z):
    return torch.tanh(z)


def act_deriv(z):
    t = torch.tanh(z)
    return 1.0 - t * t


class PredictiveCodingNet:
    def __init__(self, dims, relax_steps=20, relax_lr=0.15, weight_lr=0.08, seed=0,
                 adam=True, beta1=0.9, beta2=0.999, eps=1e-8):
        """dims: [in, hidden1, ..., out]. Всё в чистых тензорах, requires_grad=False.
        adam=True: тот же Adam, что и у backprop-baseline, но применяется к локально
        вычисленному градиенту (outer product ошибки и активности) — это оптимизатор,
        не backprop; сам градиент по-прежнему не требует .backward()."""
        g = torch.Generator().manual_seed(seed)
        self.dims = dims
        self.L = len(dims) - 1
        self.relax_steps = relax_steps
        self.relax_lr = relax_lr
        self.weight_lr = weight_lr
        self.adam = adam
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.t = 0
        self.W = [
            torch.randn(dims[l + 1], dims[l], generator=g) * (1.0 / dims[l] ** 0.5)
            for l in range(self.L)
        ]
        self.b = [torch.zeros(dims[l + 1]) for l in range(self.L)]
        if adam:
            self.mW = [torch.zeros_like(w) for w in self.W]
            self.vW = [torch.zeros_like(w) for w in self.W]
            self.mb = [torch.zeros_like(bb) for bb in self.b]
            self.vb = [torch.zeros_like(bb) for bb in self.b]

    def _adam_step(self, param, grad, m, v):
        m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param += self.weight_lr * m_hat / (v_hat.sqrt() + self.eps)

    @torch.no_grad()
    def forward_pass(self, x0):
        """Чистый feedforward прогон через выученные веса — для inference.
        Линейный (без tanh) выход на последнем слое, чтобы совпадать с MSE-целью."""
        x = x0
        for l in range(self.L):
            z = x @ self.W[l].T + self.b[l]
            x = act(z) if l < self.L - 1 else z
        return x

    @torch.no_grad()
    def train_step(self, x0, target):
        """Один шаг: forward-инициализация -> релаксация -> локальное обновление весов.
        Ни одного .backward() — структурно невозможно внутри torch.no_grad()."""
        B = x0.shape[0]

        # 1. Feedforward-инициализация свободных слоёв (это не обучение, просто
        #    хорошая стартовая точка для релаксации — ускоряет сходимость).
        xs = [x0]
        x = x0
        for l in range(self.L):
            z = x @ self.W[l].T + self.b[l]
            x = act(z) if l < self.L - 1 else z
            xs.append(x)

        xs[-1] = target  # верхний слой зажимается целью на время обучения

        # 2. Релаксация свободных слоёв (1..L-1) к энергетическому равновесию
        for _ in range(self.relax_steps):
            zs = [None] + [xs[l] @ self.W[l].T + self.b[l] for l in range(self.L)]
            eps = [None] + [xs[l + 1] - (act(zs[l + 1]) if l < self.L - 1 else zs[l + 1])
                             for l in range(self.L)]
            for l in range(1, self.L):
                # dF/dx_l = eps_l - W_{l+1}^T (eps_{l+1} * f'(z_{l+1}))
                upstream = (eps[l + 1] * act_deriv(zs[l + 1])) @ self.W[l]
                grad = eps[l] - upstream
                xs[l] = xs[l] - self.relax_lr * grad

        # 3. Локальное обновление весов: grad_W_l = (eps_l * f'(z_l)) outer x_{l-1}
        self.t += 1
        zs = [None] + [xs[l] @ self.W[l].T + self.b[l] for l in range(self.L)]
        for l in range(self.L):
            z = zs[l + 1]
            eps_l = xs[l + 1] - (act(z) if l < self.L - 1 else z)
            delta = eps_l if l == self.L - 1 else eps_l * act_deriv(z)
            gW = (delta.T @ xs[l]) / B
            gb = delta.mean(dim=0)
            if self.adam:
                self._adam_step(self.W[l], gW, self.mW[l], self.vW[l])
                self._adam_step(self.b[l], gb, self.mb[l], self.vb[l])
            else:
                self.W[l] += self.weight_lr * gW
                self.b[l] += self.weight_lr * gb

        energy = sum((e.pow(2).sum() for e in eps if e is not None)).item() / B
        return energy

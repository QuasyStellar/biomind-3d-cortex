"""
Растущий субстрат с нуля — единый "геном" (одна локальная функция на все
клетки), рост (нейрогенез) и смерть (апоптоз) выводятся из ОДНОГО и того
же сигнала локальной активности/ошибки, как в spatial_embodied_tissue.py
из v1 — но переписано заново, не скопировано, и с явным флагом для
честного A/B теста "рост включён vs выключен" под идентичным повреждением.

ВАЖНО (ограничение этой итерации, зафиксировано явно): здесь веса генома
НЕ обучаются — тестируется только структурная динамика (рост/смерть/
восстановление), не то, чему ткань "научилась". Обучение (PC-релаксация
из predictive_coding.py) сливается с этим субстратом в следующей итерации,
когда тестируем компонент 2 (память) поверх растущей ткани.
"""
import torch
import torch.nn.functional as F


def _perception_kernels(state_dim):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / 8.0
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32) / 4.0
    ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
    f = torch.stack([ident, sobel_x, sobel_y, lap])
    return f.repeat(state_dim, 1, 1).unsqueeze(1)


class GrowingTissue:
    def __init__(self, size=24, state_dim=16, seed=0,
                 neurogenesis_threshold=0.5, apoptosis_threshold=0.05,
                 dt=0.4, growth_enabled=True,
                 homeostatic=False, ema_decay=0.05, growth_ratio=1.4, decay_ratio=0.3):
        """homeostatic=True: пороги роста/апоптоза не константы, а BCM-подобная
        скользящая величина - EMA собственного стресса каждой клетки за её
        историю. Растёт, если стресс заметно ВЫШЕ своей же недавней нормы
        (growth_ratio), умирает, если заметно НИЖЕ (decay_ratio) - порог
        сам подстраивается под масштаб локальной активности, а не задан
        абсолютной константой, одинаковой для любой интенсивности стимула."""
        g = torch.Generator().manual_seed(seed)
        self.size = size
        self.state_dim = state_dim
        self.dt = dt
        self.growth_enabled = growth_enabled
        self.neurogenesis_threshold = neurogenesis_threshold
        self.apoptosis_threshold = apoptosis_threshold
        self.homeostatic = homeostatic
        self.ema_decay = ema_decay
        self.growth_ratio = growth_ratio
        self.decay_ratio = decay_ratio
        self.stress_ema = torch.full((1, 1, size, size), 0.15)

        self.p_kernels = _perception_kernels(state_dim)

        # Единый геном: маленькая локальная 1x1-conv функция, общая для всех клеток.
        hidden = 48
        self.w1 = torch.randn(hidden, state_dim * 4, generator=g) * (1.0 / (state_dim * 4) ** 0.5)
        self.b1 = torch.zeros(hidden)
        self.w2 = torch.zeros(state_dim, hidden)  # zero-init: новорождённые клетки стартуют стабильно
        self.b2 = torch.zeros(state_dim)

        self.state = torch.zeros(1, state_dim, size, size)
        self._seed_tissue()

    def _seed_tissue(self, radius=2):
        self.state.zero_()
        c = self.size // 2
        for y in range(self.size):
            for x in range(self.size):
                if (y - c) ** 2 + (x - c) ** 2 <= radius ** 2:
                    self.state[0, 0, y, x] = 1.0  # alive channel
                    self.state[0, 2:, y, x] = torch.randn(self.state_dim - 2) * 0.1

    def alive_mask(self):
        alive = self.state[:, 0:1] > 0.1
        can_grow = F.max_pool2d(alive.float(), 3, stride=1, padding=1) > 0.1
        return alive, can_grow

    def _genome(self, percept):
        h = F.gelu(F.conv2d(percept, self.w1.view(self.w1.shape[0], self.w1.shape[1], 1, 1), self.b1))
        out = F.conv2d(h, self.w2.view(self.w2.shape[0], self.w2.shape[1], 1, 1), self.b2)
        return out

    def step(self, sensory_signal=None):
        alive_before, can_grow = self.alive_mask()

        state = self.state.clone()
        if sensory_signal is not None:
            state[:, 2:4] += sensory_signal * alive_before.float()

        p_pad = F.pad(state, (1, 1, 1, 1), mode="circular")
        percept = F.conv2d(p_pad, self.p_kernels, groups=self.state_dim)
        delta = self._genome(percept)

        energy = torch.mean(state[:, 2:] ** 2, dim=1, keepdim=True)
        damping = 1.0 / (1.0 + energy * 0.05)
        state = torch.cat([state[:, :2], state[:, 2:] + torch.tanh(delta[:, 2:]) * self.dt * damping], dim=1)

        stress = torch.norm(state[:, 2:6], p=2, dim=1, keepdim=True)
        state[:, 1:2] = stress

        if self.homeostatic:
            # Обновляем скользящую норму только там, где ткань жива - у мёртвых
            # клеток EMA не должна дрейфовать в никуда.
            self.stress_ema = torch.where(
                alive_before,
                (1 - self.ema_decay) * self.stress_ema + self.ema_decay * stress,
                self.stress_ema,
            )
            grow_threshold = self.stress_ema * self.growth_ratio
            die_threshold = self.stress_ema * self.decay_ratio
        else:
            grow_threshold = torch.full_like(stress, self.neurogenesis_threshold)
            die_threshold = torch.full_like(stress, self.apoptosis_threshold)

        if self.growth_enabled:
            growth_signal = F.avg_pool2d((stress > grow_threshold).float(), 3, stride=1, padding=1)
            new_cells = (growth_signal > 0.15) & can_grow
            state[:, 0:1] = torch.where(new_cells, torch.ones_like(state[:, 0:1]), state[:, 0:1])

            idle = (stress < die_threshold) & alive_before
            state[:, 0:1] = torch.where(idle, state[:, 0:1] * 0.9, state[:, 0:1])

        alive_after = state[:, 0:1] > 0.1
        self.state = state * alive_after.float()
        return int(alive_after.sum().item())

    def ablate(self, fraction=0.4):
        """Убивает связный квадратный блок клеток площадью ~fraction от текущей ткани."""
        alive, _ = self.alive_mask()
        n_alive = int(alive.sum().item())
        if n_alive == 0:
            return 0
        block_side = max(1, int((n_alive * fraction) ** 0.5))
        c = self.size // 2
        half = block_side // 2
        y0, y1 = max(0, c - half), min(self.size, c - half + block_side)
        x0, x1 = max(0, c - half), min(self.size, c - half + block_side)
        killed = int(self.state[0, 0, y0:y1, x0:x1].gt(0.1).sum().item())
        self.state[:, :, y0:y1, x0:x1] = 0.0
        return killed

    def clone(self):
        import copy
        return copy.deepcopy(self)

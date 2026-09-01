"""
2D vs 3D эксперимент (CLAUDE_STRATEGIC_SPEC.md, раздел 1.4): честная 3D-
воксельная растущая ткань, реализована с нуля как отдельный, минимальный
класс (не полный `unified_organism.py` - для честного A/B нужен только
ЯДРО-субстрат: рост/апоптоз/абляция/PC-релаксация генома, БЕЗ памяти/VSA/
колонок, которые не относятся к вопросу "нужен ли объём сам по себе").

Мотивация вопроса (Mountcastle 1997, уже цитировано в проекте): эмбриональная
кора изначально ОДНОРОДНА, специализация возникает самоорганизованно, не
через ручное разделение. Вопрос - нужен ли ФИЗИЧЕСКИЙ объём (3D воксели,
3D-свёртки), или достаточно 2D-решётки с бОльшим числом химических каналов
состояния (что и есть текущая архитектура)?

Архитектурно - прямое обобщение `core/unified_organism.py` на 3 измерения:
state (1, state_dim, D, H, W) вместо (1, state_dim, H, W), 3D-ядра восприятия
(3x3x3 Sobel по x/y/z + 3D-Лапласиан, аналог 2D-ядер), circular padding по
всем 3 осям, 3D max-pool для alive-маски. Тот же PredictiveCodingNet-геном
(zero-backward), та же логика роста/апоптоза, что в 2D версии.
"""
import torch
import torch.nn.functional as F
from core.predictive_coding import PredictiveCodingNet


def _perception_kernels_3d(state_dim):
    """4 ядра 3x3x3: Sobel-x, Sobel-y, Sobel-z, 3D-Лапласиан - прямой аналог
    2D-набора (sobel_x, sobel_y, laplacian), обобщённый на третье измерение."""
    sobel1d = torch.tensor([-1.0, 0.0, 1.0])
    smooth1d = torch.tensor([1.0, 2.0, 1.0])

    def outer3(a, b, c):
        return torch.einsum("i,j,k->ijk", a, b, c)

    sobel_x = outer3(sobel1d, smooth1d, smooth1d) / 16.0
    sobel_y = outer3(smooth1d, sobel1d, smooth1d) / 16.0
    sobel_z = outer3(smooth1d, smooth1d, sobel1d) / 16.0

    lap = torch.zeros(3, 3, 3)
    lap[1, 1, 1] = -6.0
    for d in [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
        lap[d] = 1.0
    lap = lap / 6.0

    f = torch.stack([sobel_x, sobel_y, sobel_z, lap])  # (4, 3, 3, 3)
    return f.repeat(state_dim, 1, 1, 1, 1).view(state_dim * 4, 1, 3, 3, 3)


class LivingTissue3D:
    def __init__(self, size=8, state_dim=16, seed=0, dt=0.4, growth_enabled=True,
                 ema_decay=0.05, growth_ratio=1.4, decay_ratio=0.3,
                 genome_hidden=48, relax_steps=15, relax_lr=0.08, weight_lr=0.01,
                 growth_percentile=85.0):
        """size: сторона КУБА (size^3 воксилей) - для сопоставимости с 2D
        (size_2d^2 клеток) размер куба подбирается так, чтобы size^3 было
        близко к size_2d^2 (честное сравнение "по числу вычислительных
        единиц", не по стороне решётки - иначе 3D тривиально получает на
        порядки больше воксилей и сравнение нечестное)."""
        g = torch.Generator().manual_seed(seed)
        self.size = size
        self.state_dim = state_dim
        self.dt = dt
        self.growth_enabled = growth_enabled
        self.ema_decay, self.growth_ratio, self.decay_ratio = ema_decay, growth_ratio, decay_ratio
        self.growth_percentile = growth_percentile
        self.growth_ema_decay = 0.01
        self.growth_period = 10
        self._step_count = 0

        self.stress_ema = torch.full((1, 1, size, size, size), 0.15)
        self.growth_ema = torch.full((1, 1, size, size, size), 0.15)
        self.ctx_kernels = _perception_kernels_3d(state_dim)

        self.genome = PredictiveCodingNet(
            [state_dim * 4, genome_hidden, state_dim],
            relax_steps=relax_steps, relax_lr=relax_lr, weight_lr=weight_lr,
            seed=seed, adam=True, weight_decay=0.02)

        self.state = torch.zeros(1, state_dim, size, size, size)
        self._seed_tissue(g)

    def _seed_tissue(self, g, radius=1):
        c = self.size // 2
        for z in range(self.size):
            for y in range(self.size):
                for x in range(self.size):
                    if (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= radius ** 2:
                        self.state[0, 0, z, y, x] = 1.0
                        self.state[0, 2:, z, y, x] = torch.randn(self.state_dim - 2, generator=g) * 0.1

    def alive_mask(self):
        alive = self.state[:, 0:1] > 0.1
        can_grow = F.max_pool3d(alive.float(), 3, stride=1, padding=1) > 0.1
        return alive, can_grow

    def ablate(self, fraction=0.4):
        """Кубическая (объёмная) абляция - честный 3D-аналог квадратной 2D
        абляции: удаляет кубический РЕГИОН, не просто fraction ото всех
        клеток равномерно - как и в 2D `ablate()`, повреждение локально."""
        alive, _ = self.alive_mask()
        n_alive = int(alive.sum().item())
        if n_alive == 0:
            return 0
        block_side = max(1, int((n_alive * fraction) ** (1 / 3)))
        c = self.size // 2
        half = block_side // 2
        z0, z1 = max(0, c - half), min(self.size, c - half + block_side)
        y0, y1 = max(0, c - half), min(self.size, c - half + block_side)
        x0, x1 = max(0, c - half), min(self.size, c - half + block_side)
        killed = int(self.state[0, 0, z0:z1, y0:y1, x0:x1].gt(0.1).sum().item())
        self.state[:, :, z0:z1, y0:y1, x0:x1] = 0.0
        return killed

    def step(self, sensory_signal=None, train_genome=True):
        alive_before, can_grow = self.alive_mask()
        state = self.state.clone()
        if sensory_signal is not None:
            state[:, 2:4] += sensory_signal * alive_before.float()

        p_pad = F.pad(state, (1, 1, 1, 1, 1, 1), mode="circular")
        ctx = F.conv3d(p_pad, self.ctx_kernels, groups=self.state_dim)  # (1, 4*state_dim, D,H,W)

        zs, ys, xs = torch.where(alive_before[0, 0])
        n_alive = zs.shape[0]
        if n_alive == 0:
            return 0, 0.0

        ctx_flat = ctx[0, :, zs, ys, xs].T
        target_flat = state[0, :, zs, ys, xs].T

        pred = self.genome.train_step(ctx_flat, target_flat) if train_genome else None
        pred_out = self.genome.forward_pass(ctx_flat)
        error = target_flat - pred_out
        error_norm = error.norm(dim=1)

        new_chem = state[0, 2:, zs, ys, xs].T + self.dt * (pred_out[:, 2:] - state[0, 2:, zs, ys, xs].T)
        state[0, 2:, zs, ys, xs] = new_chem.T

        stress_map = torch.zeros(1, 1, self.size, self.size, self.size)
        stress_map[0, 0, zs, ys, xs] = error_norm
        state[:, 1:2] = stress_map

        alive_f = alive_before.float()
        self.stress_ema = torch.where(alive_before, (1 - self.ema_decay) * self.stress_ema + self.ema_decay * stress_map, self.stress_ema)
        self.growth_ema = torch.where(alive_before, (1 - self.growth_ema_decay) * self.growth_ema + self.growth_ema_decay * stress_map, self.growth_ema)
        die_th = self.stress_ema * self.decay_ratio
        self._step_count += 1
        evaluate_growth = self.growth_enabled and (self._step_count % self.growth_period == 0)

        if evaluate_growth:
            alive_growth_ema = self.growth_ema[alive_before]
            if alive_growth_ema.numel() >= 4:
                base_th = torch.quantile(alive_growth_ema, self.growth_percentile / 100.0)
            else:
                base_th = alive_growth_ema.mean() if alive_growth_ema.numel() > 0 else torch.tensor(1.0)
            growth_signal = F.avg_pool3d((self.growth_ema > base_th).float(), 3, stride=1, padding=1)
            # ИСПРАВЛЕНО ДВАЖДЫ (rule 4 - население застряло РОВНО на исходном
            # посеве на всех 5 seed, слишком чисто для архитектурного факта).
            # Первая попытка (0.15*9/27=0.05) не помогла: при малом АБСОЛЮТНОМ
            # числе "горячих" клеток (часто ровно 1 - growth_percentile=85%
            # от 7 живых даёт 1 клетку) даже идеально расположенная одна
            # горячая клетка даёt ровно 1/27=0.037 в 3x3x3-окне - ниже 0.05.
            # Порог снижен до значения, достижимого ОДНОЙ горячей соседкой
            # (аналог 2D, где 1/9=0.111 > 2D-порога 0.15 на самом деле НЕ
            # достаточно тоже - но там обычно >=2 горячих клетки одновременно
            # из-за бОльшего стартового населения 13 против 7 здесь).
            candidates = (growth_signal > 0.03) & can_grow & ~alive_before
            state[:, 0:1] = torch.where(candidates | alive_before, torch.ones_like(state[:, 0:1]), state[:, 0:1])
        if self.growth_enabled:
            idle = (stress_map < die_th) & alive_before
            state[:, 0:1] = torch.where(idle, state[:, 0:1] * 0.9, state[:, 0:1])

        alive_after = state[:, 0:1] > 0.1
        self.state = state * alive_after.float()
        return int(alive_after.sum().item()), float(error_norm.mean().item())

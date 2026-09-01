"""
Первое настоящее слияние компонентов, а не отдельные скрипты:

- Структура (рост/апоптоз) — из growing_tissue.py, НО стресс теперь
  реальная ошибка предсказания, а не просто норма активности.
- Обучение генома — PC-релаксация (predictive_coding.py), zero backward,
  self-supervised: каждая клетка предсказывает СВОЁ состояние по соседям
  (identity-канал предсказывается из sobel/laplacian-контекста) - тот же
  принцип "замаскированный токен из контекста", что в JEPA-тесте, только
  в пространстве, а не в тексте.
- Гомеостаз (BCM) — самонастраивающиеся пороги роста/апоптоза поверх
  ЭТОГО ЖЕ сигнала ошибки предсказания.
- Быстрая память (W_fast) — общая пластичная матрица, симметричное
  Хеббовское правило (эволюция ABCD не прижилась - используем ручное,
  провал зафиксирован в VERIFICATION_LOG.md), запись С ПРИОРИТЕТОМ по
  силе метки (synaptic tagging and capture, Redondo & Morris) - тег
  силы записи пропорционален ошибке предсказания/новизне В МОМЕНТ
  записи, а не одинаков для всех фактов.
"""
import torch
import torch.nn.functional as F
from core.predictive_coding import PredictiveCodingNet


def _perception_kernels(state_dim):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / 8.0
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32) / 4.0
    f = torch.stack([sobel_x, sobel_y, lap])
    return f.repeat(state_dim, 1, 1).unsqueeze(1)


class LivingTissue:
    def __init__(self, size=24, state_dim=16, seed=0, dt=0.4, growth_enabled=True,
                 ema_decay=0.05, growth_ratio=1.4, decay_ratio=0.3,
                 genome_hidden=48, relax_steps=15, relax_lr=0.08, weight_lr=0.01):
        g = torch.Generator().manual_seed(seed)
        self.size = size
        self.state_dim = state_dim
        self.dt = dt
        self.growth_enabled = growth_enabled
        self.ema_decay, self.growth_ratio, self.decay_ratio = ema_decay, growth_ratio, decay_ratio
        self.stress_ema = torch.full((1, 1, size, size), 0.15)

        self.ctx_kernels = _perception_kernels(state_dim)  # только sobel/lap - контекст соседей

        # Единый геном - предсказывает СЕБЯ (identity) по КОНТЕКСТУ (соседи).
        # Zero backward - обучается той же PC-релаксацией, что уже проверена.
        self.genome = PredictiveCodingNet(
            [state_dim * 3, genome_hidden, state_dim],
            relax_steps=relax_steps, relax_lr=relax_lr, weight_lr=weight_lr,
            seed=seed, adam=True, weight_decay=0.02)

        # Быстрая ассоциативная память - общая на всю ткань, symmetric delta-rule,
        # с SDR-разреженностью (компонент 2, уже проверено - плотная память без
        # неё деградирует быстро при N>~100, см. hippocampus_retention_sweep.py).
        # Забыл применить это в первой версии слияния - сырая плотная матрица
        # дала 8/20 и 1/20 recall, что и предсказывает уже известная находка.
        self.fast_dim = state_dim - 2
        sdr_dim = 512
        self.sdr_sparsity = 0.08
        self.sdr_k = max(1, int(sdr_dim * self.sdr_sparsity))
        self.dg_proj = torch.randn(sdr_dim, self.fast_dim, generator=g) * (1.0 / self.fast_dim ** 0.5)
        self.W_fast = torch.zeros(self.fast_dim, sdr_dim)

        self.state = torch.zeros(1, state_dim, size, size)
        self._seed_tissue()

    def _sdr_code(self, key):
        h = torch.relu(self.dg_proj @ key)
        val, idx = torch.topk(h, self.sdr_k)
        sdr = torch.zeros_like(h)
        sdr[idx] = val
        return sdr / (sdr.norm() + 1e-7)

    def _seed_tissue(self, radius=2):
        self.state.zero_()
        c = self.size // 2
        for y in range(self.size):
            for x in range(self.size):
                if (y - c) ** 2 + (x - c) ** 2 <= radius ** 2:
                    self.state[0, 0, y, x] = 1.0
                    self.state[0, 2:, y, x] = torch.randn(self.state_dim - 2) * 0.1

    def alive_mask(self):
        alive = self.state[:, 0:1] > 0.1
        can_grow = F.max_pool2d(alive.float(), 3, stride=1, padding=1) > 0.1
        return alive, can_grow

    def step(self, sensory_signal=None, train_genome=True):
        alive_before, can_grow = self.alive_mask()
        state = self.state.clone()
        if sensory_signal is not None:
            state[:, 2:4] += sensory_signal * alive_before.float()

        p_pad = F.pad(state, (1, 1, 1, 1), mode="circular")
        ctx = F.conv2d(p_pad, self.ctx_kernels, groups=self.state_dim)  # (1, 3*state_dim, H, W)

        ys, xs = torch.where(alive_before[0, 0])
        n_alive = ys.shape[0]
        if n_alive == 0:
            return 0, 0.0

        ctx_flat = ctx[0, :, ys, xs].T  # (n_alive, 3*state_dim)

        # genome выдаёт предсказание размерности state_dim (полное), сравниваем
        # с полным state (включая alive/stress channel 0:2 как есть, без
        # прогноза по ним - берём state[0,:,ys,xs] целиком для простоты формы).
        target_flat = state[0, :, ys, xs].T  # (n_alive, state_dim)

        if train_genome:
            pred_error_energy = self.genome.train_step(ctx_flat, target_flat)
        pred = self.genome.forward_pass(ctx_flat)  # (n_alive, state_dim)
        error = target_flat - pred  # РЕАЛЬНАЯ ошибка предсказания на клетку
        error_norm = error.norm(dim=1)  # (n_alive,)

        # Обновляем chemistry - релаксация к тому, что контекст предсказывает
        # (движение к консенсусу соседей), плюс небольшой вклад W_fast.
        new_chem = state[0, 2:, ys, xs].T + self.dt * (pred[:, 2:] - state[0, 2:, ys, xs].T)
        state[0, 2:, ys, xs] = new_chem.T

        # Реальный стресс = ошибка предсказания (не просто норма активности,
        # как раньше) - совпадает с дизайном morphogenetic_3d_cortex.py из
        # архива, который мы раньше признали архитектурно самым честным.
        stress_map = torch.zeros(1, 1, self.size, self.size)
        stress_map[0, 0, ys, xs] = error_norm
        state[:, 1:2] = stress_map

        # BCM-гомеостаз - ТОЛЬКО для апоптоза. Найдено (после двух неудачных
        # попыток - ограничение скорости и калибровка при рождении): для
        # роста self-referential порог структурно нестабилен, потому что
        # новорождённая клетка не имеет собственной истории ошибки, а любой
        # способ её "изначально откалибровать" либо не успевает за реальной
        # (высокой, т.к. геном не знает новый контекст) ошибкой, либо
        # обнуляет порог и делает рост ЕЩЁ агрессивнее. Апоптоз безопасен -
        # там EMA берётся у УЖЕ проживших хотя бы шаг клеток, без bootstrap-
        # проблемы. Рост использует фиксированный порог 1.6 (выше пикового
        # наблюдаемого раннего error ~1.46) - подобран экспериментально по
        # трассировке (0.5 давало заливку всего холста за 20 шагов; 1.6
        # даёт ограниченный, но не остановленный рост 13->30->85 за 150
        # шагов). Это временное решение первой версии слияния, не финальная
        # калибровка - самонастройка порога РОСТА (не только апоптоза) на
        # основе ошибки предсказания остаётся открытой задачей.
        self.stress_ema = torch.where(alive_before, (1 - self.ema_decay) * self.stress_ema + self.ema_decay * stress_map, self.stress_ema)
        die_th = self.stress_ema * self.decay_ratio
        grow_th_fixed = 1.6

        if self.growth_enabled:
            growth_signal = F.avg_pool2d((stress_map > grow_th_fixed).float(), 3, stride=1, padding=1)
            candidates = (growth_signal > 0.15) & can_grow & ~alive_before
            state[:, 0:1] = torch.where(candidates | alive_before, torch.ones_like(state[:, 0:1]), state[:, 0:1])
            idle = (stress_map < die_th) & alive_before
            state[:, 0:1] = torch.where(idle, state[:, 0:1] * 0.9, state[:, 0:1])

        alive_after = state[:, 0:1] > 0.1
        self.state = state * alive_after.float()
        return int(alive_after.sum().item()), float(error_norm.mean().item())

    # --- Быстрая память: приоритет по силе метки (synaptic tagging) ---
    def write_fact(self, key, value, tag_strength=1.0, beta=0.9):
        """tag_strength - сила метки (0..~2), пропорциональна новизне/ошибке
        в момент записи (Redondo & Morris, synaptic tagging and capture) -
        не все факты пишутся с одинаковой силой, как было раньше."""
        s = self._sdr_code(key)
        pred = self.W_fast @ s
        err = value - pred
        self.W_fast += beta * tag_strength * torch.outer(err, s)

    def read_fact(self, key):
        return self.W_fast @ self._sdr_code(key)

    def ablate(self, fraction=0.4):
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

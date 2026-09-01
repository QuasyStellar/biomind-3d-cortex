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
                 genome_hidden=48, relax_steps=15, relax_lr=0.08, weight_lr=0.01,
                 replay_capacity=4000, replay_batch=192, replay_min_before_train=192,
                 predict_chem_only=False, si_enabled=False, si_lambda=1.0):
        """replay_*: буфер прошлых (контекст, цель) пар вместо обучения только
        на срезе текущего шага - без этого геном не сходится (найдено:
        error 1.07->1.19 за 300 шагов растущей ткани), потому что растущая
        популяция клеток - постоянно меняющаяся, нестационарная обучающая
        выборка. Тот же принцип, что и rehearsal-реплей в компоненте 3 (сон)."""
        g = torch.Generator().manual_seed(seed)
        self.replay_capacity = replay_capacity
        self.replay_batch = replay_batch
        self.replay_min_before_train = replay_min_before_train
        self._replay_ctx = None
        self._replay_target = None
        self._replay_ptr = 0
        self._replay_full = False
        self._step_count = 0
        # Рост оценивается по СГЛАЖЕННОМУ (много медленнее, чем апоптоз) сигналу
        # ошибки и применяется раз в growth_period шагов - развязка по скорости
        # с обучением весов (каждый шаг), а не разделение на дискретные фазы.
        # В реальном мозге структурная пластичность на порядки медленнее
        # синаптической - это и моделируем, а не изобретаем произвольный костыль.
        self.growth_ema_decay = 0.01
        self.growth_period = 10
        self.growth_percentile = 85.0  # относительный, не абсолютный порог - см. step()
        self.growth_ema = torch.full((1, 1, size, size), 0.15)
        # "Воспаление" - временный, локальный сигнал вокруг свежей раны,
        # понижающий порог роста именно там и затухающий со временем.
        # Не глобальная и не постоянная поправка (иначе вернём разгон) -
        # реальное заживление раны тоже работает через временную локальную
        # сенсибилизацию к факторам роста, не через постоянно другой порог.
        self.inflammation = torch.zeros(1, 1, size, size)
        self.inflammation_decay = 0.96
        self.inflammation_threshold_drop = 0.9  # доля понижения порога в пике воспаления
        self.size = size
        self.state_dim = state_dim
        self.dt = dt
        self.growth_enabled = growth_enabled
        self.ema_decay, self.growth_ratio, self.decay_ratio = ema_decay, growth_ratio, decay_ratio
        self.stress_ema = torch.full((1, 1, size, size), 0.15)

        self.ctx_kernels = _perception_kernels(state_dim)  # только sobel/lap - контекст соседей

        # Единый геном - предсказывает СЕБЯ (identity) по КОНТЕКСТУ (соседи).
        # Zero backward - обучается той же PC-релаксацией, что уже проверена.
        self.predict_chem_only = predict_chem_only
        out_dim = (state_dim - 2) if predict_chem_only else state_dim
        self.genome = PredictiveCodingNet(
            [state_dim * 3, genome_hidden, out_dim],
            relax_steps=relax_steps, relax_lr=relax_lr, weight_lr=weight_lr,
            seed=seed, adam=True, weight_decay=0.02,
            si_enabled=si_enabled, si_lambda=si_lambda)

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

        # Мультимодальная быстрая память (мердж columnar_voting.py в организм,
        # приоритет (a) README/ROADMAP): каждая "колонка" - своя независимая
        # SDR-память НА СВОЙ ключ-домен (одна и та же цель может быть записана
        # разными сенсорными путями - например, из контекста ткани И из внешнего
        # символьного ключа). Дефолтная W_fast/dg_proj выше остаются как есть
        # (write_fact/read_fact - обратная совместимость с unified_organism_sanity.py),
        # add_modality() добавляет ДОПОЛНИТЕЛЬНЫЕ независимые колонки.
        self.modalities = {}

        self.state = torch.zeros(1, state_dim, size, size)
        self._seed_tissue()

    def _sdr_code(self, key, dg_proj=None, sdr_k=None):
        proj = self.dg_proj if dg_proj is None else dg_proj
        k = self.sdr_k if sdr_k is None else sdr_k
        h = torch.relu(proj @ key)
        val, idx = torch.topk(h, k)
        sdr = torch.zeros_like(h)
        sdr[idx] = val
        return sdr / (sdr.norm() + 1e-7)

    def add_modality(self, name, key_dim, sdr_dim=512, sparsity=0.08, seed=None, value_dim=None):
        """Регистрирует независимую SDR-колонку для модальности `name`
        (свой dg_proj под свою размерность ключа, свой W - НЕ общий с
        default/другими модальностями, как в Column из columnar_voting.py).
        value_dim по умолчанию = fast_dim (химия клетки), но можно задать
        другую размерность значения - нужно, например, для flat-baseline
        в unified_vsa_compositional_sanity.py, где значения - VSA-векторы
        произвольной размерности, не привязанные к химии ткани."""
        g = torch.Generator().manual_seed(seed if seed is not None else abs(hash(name)) % (2 ** 31))
        sdr_k = max(1, int(sdr_dim * sparsity))
        self.modalities[name] = {
            "dg_proj": torch.randn(sdr_dim, key_dim, generator=g) * (1.0 / key_dim ** 0.5),
            "W": torch.zeros(value_dim if value_dim is not None else self.fast_dim, sdr_dim),
            "sdr_k": sdr_k,
        }

    def write_fact_modal(self, modality, key, value, tag_strength=1.0, beta=0.9):
        m = self.modalities[modality]
        s = self._sdr_code(key, m["dg_proj"], m["sdr_k"])
        pred = m["W"] @ s
        err = value - pred
        m["W"] += beta * tag_strength * torch.outer(err, s)

    def read_fact_modal(self, modality, key):
        m = self.modalities[modality]
        return m["W"] @ self._sdr_code(key, m["dg_proj"], m["sdr_k"])

    def read_fact_voted(self, keys_by_modality):
        """Слияние ЧТЕНИЙ нескольких колонок без ручных весов на модальность -
        тот же принцип, что vote_consensus в columnar_voting.py (там - сумма
        нормированных косинусных голосов по дискретным кандидатам), здесь
        обобщено на НЕПРЕРЫВНОЕ значение: каждая активная модальность "голосует"
        своим предсказанием, вес голоса = уверенность (норма предсказания) ЭТОЙ
        колонки, нормированная по сумме - колонка с "пустой"/незаписанной
        памятью (низкая норма) естественно вносит меньший вклад, без constant
        per-modality весов, которые ломаются при добавлении новой модальности
        (та же находка, что и в columnar_voting_sanity.py)."""
        preds, weights = [], []
        for name, key in keys_by_modality.items():
            if name not in self.modalities:
                continue
            p = self.read_fact_modal(name, key)
            preds.append(p)
            weights.append(p.norm())
        if not preds:
            return torch.zeros(self.fast_dim)
        w = torch.stack(weights)
        if float(w.sum()) < 1e-7:
            return torch.zeros_like(preds[0])
        wn = w / w.sum()
        return sum(p * wi for p, wi in zip(preds, wn))

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

    def _replay_push(self, ctx_flat, target_flat):
        # device=ctx_flat.device - без этого буфер всегда создавался бы на CPU
        # даже если вся остальная ткань на CUDA, и падал бы с device mismatch
        # при первой же записи (найдено на Colab при повторных абляциях).
        ctx_dim, target_dim = ctx_flat.shape[1], target_flat.shape[1]
        if self._replay_ctx is None:
            self._replay_ctx = torch.zeros(self.replay_capacity, ctx_dim, device=ctx_flat.device)
            self._replay_target = torch.zeros(self.replay_capacity, target_dim, device=ctx_flat.device)
        n = ctx_flat.shape[0]
        if n >= self.replay_capacity:
            idx = torch.randperm(n, device=ctx_flat.device)[: self.replay_capacity]
            self._replay_ctx[:] = ctx_flat[idx].detach()
            self._replay_target[:] = target_flat[idx].detach()
            self._replay_ptr = 0
            self._replay_full = True
            return
        end = self._replay_ptr + n
        if end <= self.replay_capacity:
            self._replay_ctx[self._replay_ptr:end] = ctx_flat.detach()
            self._replay_target[self._replay_ptr:end] = target_flat.detach()
        else:
            first = self.replay_capacity - self._replay_ptr
            self._replay_ctx[self._replay_ptr:] = ctx_flat[:first].detach()
            self._replay_target[self._replay_ptr:] = target_flat[:first].detach()
            self._replay_ctx[: n - first] = ctx_flat[first:].detach()
            self._replay_target[: n - first] = target_flat[first:].detach()
            self._replay_full = True
        self._replay_ptr = end % self.replay_capacity
        if end >= self.replay_capacity:
            self._replay_full = True

    def _replay_sample(self, batch_size):
        limit = self.replay_capacity if self._replay_full else self._replay_ptr
        if limit == 0:
            return None, None
        idx = torch.randint(0, limit, (min(batch_size, limit),), device=self._replay_ctx.device)
        return self._replay_ctx[idx], self._replay_target[idx]

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

        # target: либо полный state (включая alive/stress "бухгалтерские"
        # каналы 0:2), либо только chemistry (2:) при predict_chem_only=True -
        # гипотеза: alive/stress добавляют нередуцируемый шум в цель предсказания.
        if self.predict_chem_only:
            target_flat = state[0, 2:, ys, xs].T  # (n_alive, state_dim-2)
        else:
            target_flat = state[0, :, ys, xs].T  # (n_alive, state_dim)

        self._replay_push(ctx_flat, target_flat)
        if train_genome:
            train_ctx, train_target = self._replay_sample(self.replay_batch)
            if train_ctx is not None and train_ctx.shape[0] >= self.replay_min_before_train:
                pred_error_energy = self.genome.train_step(train_ctx, train_target)
        pred = self.genome.forward_pass(ctx_flat)  # (n_alive, state_dim)
        error = target_flat - pred  # РЕАЛЬНАЯ ошибка предсказания на клетку
        error_norm = error.norm(dim=1)  # (n_alive,)

        # Обновляем chemistry - релаксация к тому, что контекст предсказывает
        # (движение к консенсусу соседей), плюс небольшой вклад W_fast.
        pred_chem = pred if self.predict_chem_only else pred[:, 2:]
        new_chem = state[0, 2:, ys, xs].T + self.dt * (pred_chem - state[0, 2:, ys, xs].T)
        state[0, 2:, ys, xs] = new_chem.T

        # Реальный стресс = ошибка предсказания (не просто норма активности,
        # как раньше) - совпадает с дизайном morphogenetic_3d_cortex.py из
        # архива, который мы раньше признали архитектурно самым честным.
        # device=state.device - без этого падает с device mismatch на CUDA
        # (найдено на Colab при повторных абляциях - раньше не всплывало,
        # т.к. первый Colab-тест организма был короче одного цикла).
        stress_map = torch.zeros(1, 1, self.size, self.size, device=state.device)
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
        # проблемы, использует быстрый ema_decay=0.05.
        #
        # ПЕРЕСМОТРЕНО по вопросу "не костыль ли wake/sleep-разделение?":
        # реальная причина нестабильности - рост и обучение весов сидят на
        # ОДНОМ тике, поэтому решение о делении клетки принимается по
        # мгновенной, шумной ошибке ОДНОГО шага. В мозге структурная
        # пластичность на порядки медленнее синаптической - разводим по
        # СКОРОСТИ (медленная EMA + решение о росте раз в growth_period
        # шагов), а не по дискретным day/night фазам, что было бы
        # изобретённым под конкретный баг решением, а не биологически
        # обоснованным.
        self.stress_ema = torch.where(alive_before, (1 - self.ema_decay) * self.stress_ema + self.ema_decay * stress_map, self.stress_ema)
        self.growth_ema = torch.where(alive_before, (1 - self.growth_ema_decay) * self.growth_ema + self.growth_ema_decay * stress_map, self.growth_ema)
        die_th = self.stress_ema * self.decay_ratio
        self._step_count += 1
        evaluate_growth = self.growth_enabled and (self._step_count % self.growth_period == 0)

        # Воспаление затухает каждый шаг (временное, не постоянное).
        self.inflammation = self.inflammation * self.inflammation_decay

        if evaluate_growth:
            # PERCENTILE-порог вместо абсолютной константы (найдено на Colab:
            # константа 1.7, откалиброванная на холсте 24x24, вообще не
            # запускала рост на 128x128 - фиксированные числа не масштабируются).
            # Порог = N-й процентиль РЕАЛЬНОГО распределения growth_ema среди
            # ЖИВЫХ клеток прямо сейчас - самонастраивается под любой масштаб
            # и любую статистику ошибки автоматически, без магической константы.
            # Обходит и bootstrap-проблему новорождённых: они не участвуют в
            # вычислении процентиля (используются только alive_before клетки),
            # и не нуждаются в собственной истории для сравнения.
            alive_growth_ema = self.growth_ema[alive_before]
            if alive_growth_ema.numel() >= 4:
                base_th = torch.quantile(alive_growth_ema, self.growth_percentile / 100.0)
            else:
                base_th = alive_growth_ema.mean() if alive_growth_ema.numel() > 0 else torch.tensor(1.0, device=self.growth_ema.device)
            local_grow_th = base_th * (1.0 - self.inflammation_threshold_drop * self.inflammation)
            growth_signal = F.avg_pool2d((self.growth_ema > local_grow_th).float(), 3, stride=1, padding=1)
            candidates = (growth_signal > 0.15) & can_grow & ~alive_before
            state[:, 0:1] = torch.where(candidates | alive_before, torch.ones_like(state[:, 0:1]), state[:, 0:1])
        if self.growth_enabled:
            idle = (stress_map < die_th) & alive_before
            state[:, 0:1] = torch.where(idle, state[:, 0:1] * 0.9, state[:, 0:1])

        alive_after = state[:, 0:1] > 0.1
        self.state = state * alive_after.float()
        return int(alive_after.sum().item()), float(error_norm.mean().item())

    # --- VSA-связывание, слито из vsa_binding.py (приоритет a): роль-заполнитель
    # пары связываются через circular convolution (Plate 1995, ёмкостная кривая
    # уже проверена отдельно до K=512 на dim=1024), суперпозиция нескольких пар
    # пишется В SDR-память ОДНОЙ записью с приоритетом по метке (synaptic
    # tagging, тоже уже проверено отдельно) - ВПЕРВЫЕ совмещены здесь, каждый
    # компонент раньше тестировался только сам по себе. Значение bundle живёт
    # в СВОЁМ пространстве (vsa_dim), не в fast_dim ткани (fast_dim=state_dim-2
    # слишком мал для интересной ёмкости VSA) - отдельная W-матрица, не путать
    # с add_modality (та привязана к fast_dim). ---
    def init_vsa(self, vsa_dim=256, n_roles=16, sdr_dim=512, sparsity=0.08, seed=0):
        from core.vsa_binding import random_vectors
        self.vsa_dim = vsa_dim
        self.vsa_roles = random_vectors(n_roles, vsa_dim, seed=seed)
        g = torch.Generator().manual_seed(seed + 1)
        self.vsa_sdr_k = max(1, int(sdr_dim * sparsity))
        self.vsa_dg_proj = torch.randn(sdr_dim, vsa_dim, generator=g) * (1.0 / vsa_dim ** 0.5)
        self.vsa_W = torch.zeros(vsa_dim, sdr_dim)

    def write_compositional(self, slot_key, role_filler_pairs, tag_strength=1.0, beta=0.9):
        """slot_key: ключ "объекта" (например, вектор сущности). role_filler_pairs:
        [(role_idx, filler_vec), ...] - связываются через circular convolution и
        суперпозируются в ОДИН bundle, пишется ОДНОЙ SDR-записью (не N отдельных)."""
        from core.vsa_binding import circular_conv
        bundle = torch.zeros(self.vsa_dim)
        for role_idx, filler in role_filler_pairs:
            bundle = bundle + circular_conv(self.vsa_roles[role_idx], filler)
        s = self._sdr_code(slot_key, self.vsa_dg_proj, self.vsa_sdr_k)
        pred = self.vsa_W @ s
        err = bundle - pred
        self.vsa_W += beta * tag_strength * torch.outer(err, s)

    def read_compositional(self, slot_key, role_idx):
        from core.vsa_binding import circular_corr
        bundle = self.vsa_W @ self._sdr_code(slot_key, self.vsa_dg_proj, self.vsa_sdr_k)
        return circular_corr(bundle, self.vsa_roles[role_idx])

    # --- JEPA-понимание, слито из jepa_understanding_sanity.py (приоритет a):
    # геном уже обучается self-supervised предсказанию себя по контексту
    # (тот же принцип, что маскированный токен в тексте) - его СКРЫТОЕ
    # представление доступно снаружи для downstream-задач, которым геном
    # никогда явно не обучался, без дублирования логики вычисления context. ---
    def compute_context(self):
        """(ctx_flat, ys, xs) для ТЕКУЩЕГО состояния - тот же контекст,
        что скармливается геному в step(), доступный отдельно для внешнего
        анализа представлений."""
        alive, _ = self.alive_mask()
        p_pad = F.pad(self.state, (1, 1, 1, 1), mode="circular")
        ctx = F.conv2d(p_pad, self.ctx_kernels, groups=self.state_dim)
        ys, xs = torch.where(alive[0, 0])
        ctx_flat = ctx[0, :, ys, xs].T
        return ctx_flat, ys, xs

    def hidden_representation(self, ctx_flat):
        """Скрытое представление первого слоя генома - self-supervised
        представление (см. jepa_understanding_sanity.py: тот же приём -
        активация скрытого слоя сети, обученной предсказывать маскированную
        часть входа по остальному, полезна для downstream-классификации,
        которой сеть никогда явно не обучалась)."""
        z1 = ctx_flat @ self.genome.W[0].T + self.genome.b[0]
        return torch.tanh(z1)

    # --- Phase-binding, слито из phase_binding.py (приоритет a, последний
    # пункт из "колонки/JEPA/phase-binding"). Механизм проверен отдельно:
    # математически корректен (идеальный случай - точная сегментация r=0),
    # но качество на СЛУЧАЙНЫХ representations зависит от знака cos-сходства
    # (см. VERIFICATION_LOG). Здесь - честный вопрос: у РЕАЛЬНОЙ, обученной
    # PC-релаксацией химии ткани (не случайных векторов) естественно ли
    # возникает нужная (не положительная) корреляция между РАЗНЫМИ областями,
    # или это тоже нужно проектировать явно? ---
    def init_phase(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.phase = 2 * torch.pi * torch.rand(self.size, self.size, generator=g)

    def phase_sync_step(self, K=1.0, dt=0.1, sim_gate=True):
        """Один шаг Kuramoto-релаксации по ЖИВЫМ клеткам, используя РЕАЛЬНУЮ
        (обучаемую) химию ткани (state[0, 2:]) как основу сходства - не
        синтетические векторы, как в изолированном phase_binding_sanity.py."""
        from core.phase_binding import kuramoto_step
        if not hasattr(self, "phase"):
            self.init_phase()
        alive, _ = self.alive_mask()
        chem = self.state[0, 2:]
        self.phase = kuramoto_step(chem, self.phase, alive[0, 0], K=K, dt=dt, sim_gate=sim_gate)
        return self.phase

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
        # Воспаление помечает зону раны И небольшую кайму вокруг (+2 клетки) -
        # именно выжившие пограничные клетки должны прорастать в рану,
        # поэтому сенсибилизировать нужно их, а не только мёртвую область.
        my0, my1 = max(0, y0 - 2), min(self.size, y1 + 2)
        mx0, mx1 = max(0, x0 - 2), min(self.size, x1 + 2)
        self.inflammation[:, :, my0:my1, mx0:mx1] = 1.0
        return killed

    def clone(self):
        import copy
        return copy.deepcopy(self)

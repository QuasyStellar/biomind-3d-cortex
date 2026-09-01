"""
Второй уровень PC-релаксации над pooled-сеткой per-cell признаков - идея,
записанная (не реализованная) в docs/VERIFICATION_LOG.md, раздел
"Архитектурные изменения": сейчас между per-cell предсказаниями и финальным
признаком стоит ПЛОСКОЕ усреднение (SPATIAL_POOL) - клетки предсказывают
локально, их выходы механически усредняются в сетку без обработки ОТНОШЕНИЙ
между регионами сетки. Здесь вместо этого - вторая PredictiveCodingNet
(уже проверенный, существующий класс, НЕ новая механика релаксации) с
локальной "дендритной" маской первого слоя (w0_mask - тоже уже существующий,
проверенный параметр): каждая "супер-клетка" укрупнённой сетки видит только
3x3 контекст соседних супер-клеток по ВСЕМ каналам, не всю сетку целиком -
тот же принцип локального рецептивного поля (ctx_kernels в
unified_organism.py), примененный на уровень выше, к сетке из pool x pool
регионов вместо сырых клеток ткани (V1->V2-подобная иерархия рецептивных
полей, Mountcastle 1997, колончатая организация коры).

Self-supervised задача: узкое-место автоэнкодер (bottleneck h2_dim < h_dim
на позицию сетки) - сеть учится реконструировать СВОЙ локальный 3x3-контекст,
без учителя/меток, тем же принципом, что и сама ткань (zero-backward,
train_step из PredictiveCodingNet). Обучается ОТДЕЛЬНОЙ фазой на
baseline-вычтенном (не сыром) L1-признаке, чтобы не наследовать уже
известную причину коллапса (сырое представление почти неотличимо для
разных изображений - см. VERIFICATION_LOG, "причина №1").
"""
import torch
from core.predictive_coding import PredictiveCodingNet, act


def build_local_mask(pool, h_dim, h2_dim, radius=1):
    """mask[out_idx, in_idx] = 1, если пространственная позиция выходной
    супер-клетки (h2_dim каналов на позицию) в Чебышёвском радиусе `radius`
    от пространственной позиции входного канала (h_dim каналов на позицию),
    иначе 0. Маскируется ТОЛЬКО по пространству - все h2_dim выходных
    каналов данной позиции видят ВСЕ h_dim входных каналов каждой соседней
    позиции (включая саму себя при radius>=0)."""
    D_out = h2_dim * pool * pool
    D_in = h_dim * pool * pool
    mask = torch.zeros(D_out, D_in)
    for y in range(pool):
        for x in range(pool):
            ys, ye = max(0, y - radius), min(pool, y + radius + 1)
            xs, xe = max(0, x - radius), min(pool, x + radius + 1)
            out_idx = [h2 * pool * pool + y * pool + x for h2 in range(h2_dim)]
            in_idx = [c * pool * pool + ny * pool + nx
                      for ny in range(ys, ye) for nx in range(xs, xe)
                      for c in range(h_dim)]
            mask[torch.tensor(out_idx).unsqueeze(1), torch.tensor(in_idx).unsqueeze(0)] = 1.0
    return mask


class HierarchicalPC:
    """dims=[D, D2, D] (D=h_dim*pool*pool, D2=h2_dim*pool*pool), w0_mask на
    первом слое (кодировщик, локальный по пространству), второй слой
    (декодировщик, D2->D) полносвязный - декодер не обязан повторять
    структурное ограничение энкодера, как и в обычных свёрточных
    автоэнкодерах. Признак для downstream = `encode()` (D2-мерный
    bottleneck), не выход декодера."""

    def __init__(self, pool, h_dim, h2_dim, radius=1, seed=0, **pc_kwargs):
        self.pool = pool
        self.h_dim = h_dim
        self.h2_dim = h2_dim
        D = h_dim * pool * pool
        D2 = h2_dim * pool * pool
        mask = build_local_mask(pool, h_dim, h2_dim, radius=radius)
        self.net = PredictiveCodingNet([D, D2, D], w0_mask=mask, seed=seed, **pc_kwargs)

    def train_step(self, x_batch):
        return self.net.train_step(x_batch, x_batch)

    @torch.no_grad()
    def encode(self, x):
        """Прямой проход ТОЛЬКО до bottleneck-слоя (без релаксации,
        аналогично forward_pass) - для извлечения признака downstream без
        обновления весов."""
        z = x @ self.net.W[0].T + self.net.b[0]
        return act(z)

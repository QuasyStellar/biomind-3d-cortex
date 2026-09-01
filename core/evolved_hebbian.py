"""
Обобщённое ABCD-правило Хебба (Najarro & Risi, 2020) с нуля вместо
ручного подбора одной константы beta. Правило: Δw = A·(pre⊗target) +
B·(pre⊗pred) + C·pre + D. Наш прежний ручной delta-rule - частный случай
A=beta, B=-beta, C=D=0. Коэффициенты ищутся простой (mu,lambda)-эволюционной
стратегией собственной реализации (без внешних библиотек CMA-ES).
"""
import torch


class EvolvedHebbianHippocampus:
    """Тот же интерфейс, что SDRHippocampus, но правило записи параметризовано
    четырьмя эволюционируемыми коэффициентами вместо одной ручной beta."""
    def __init__(self, dim, sdr_dim=1024, sparsity=0.06, coeffs=(0.9, -0.9, 0.0, 0.0), seed=0):
        g = torch.Generator().manual_seed(seed)
        self.dim = dim
        self.sdr_dim = sdr_dim
        self.k = max(1, int(sdr_dim * sparsity))
        self.A, self.B, self.C, self.D = coeffs
        self.dg_proj = torch.randn(sdr_dim, dim, generator=g) * (1.0 / dim ** 0.5)
        self.W = torch.zeros(dim, sdr_dim)

    def code(self, key):
        h = torch.relu(self.dg_proj @ key)
        val, idx = torch.topk(h, self.k)
        sdr = torch.zeros_like(h)
        sdr[idx] = val
        norm = sdr.norm() + 1e-7
        return sdr / norm

    def write(self, key, value):
        s = self.code(key)
        pred = self.W @ s
        # Обобщённое ABCD-правило: A на цель, B на предсказание (вместе дают
        # error-correction при A=-B), C - пресинаптический член, D - константный
        # сдвиг. НАЙДЕНО И ИСПРАВЛЕНО: C и D применялись ко ВСЕЙ матрице
        # (dim*sdr_dim элементов) каждую запись без нормировки - даже C=D=0.1
        # накапливало чудовищное смещение за N записей и полностью убивало
        # сигнал (fitness падал с 99% до 0.7%). Нормируем на sdr_dim, чтобы
        # C/D были сопоставимы по масштабу с локальными A/B-членами, а не
        # системно доминировали над ними за счёт широковещательного broadcast.
        self.W += self.A * torch.outer(value, s) + self.B * torch.outer(pred, s) \
                  + (self.C / self.sdr_dim) * s.unsqueeze(0).expand_as(self.W) \
                  + (self.D / self.sdr_dim)

    def read(self, key):
        return self.W @ self.code(key)


def evolve_abcd(fitness_fn, generations=25, population=16, elite=4, sigma=0.3, seed=0,
                 sigma_cd=0.02):
    """sigma_cd: отдельный (более узкий) масштаб поиска для C/D - найдено,
    что при одинаковом с A/B масштабе C/D систематически ломают правило
    (широковещательный broadcast на всю матрицу даже после нормировки на
    sdr_dim остаётся гораздо чувствительнее локальных A/B-членов)."""
    """Простая (mu,lambda)-эволюционная стратегия с нуля: без CMA-ES,
    без внешних библиотек. fitness_fn(coeffs) -> float (выше = лучше).

    Найден и исправлен реальный баг: без явного элитизма (сохранения
    лучшей когда-либо найденной особи В САМОЙ популяции следующего
    поколения) поиск может преждевременно сойтись мимо хорошего решения -
    обнаружено, когда ES дала 74% там, где обычный grid-search по одной
    оси нашёл 99.3% в пределах того же пространства поиска."""
    g = torch.Generator().manual_seed(seed)
    mean = torch.tensor([0.5, -0.5, 0.0, 0.0])
    std = torch.tensor([sigma, sigma, sigma_cd, sigma_cd])

    best_coeffs, best_fit = None, -1e9
    history = []
    for gen in range(generations):
        pop = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(population, 4, generator=g)
        if best_coeffs is not None:
            pop[0] = torch.tensor(best_coeffs)  # явный элитизм - лучшая особь всегда в популяции
        fits = torch.tensor([fitness_fn(tuple(ind.tolist())) for ind in pop])
        order = torch.argsort(fits, descending=True)
        elite_pop = pop[order[:elite]]
        mean = elite_pop.mean(dim=0)
        min_std = torch.tensor([sigma, sigma, sigma_cd, sigma_cd]) * 0.25  # per-dim пол, не даём застрять
        std = elite_pop.std(dim=0).clamp(min=min_std)

        gen_best_fit = fits[order[0]].item()
        gen_best = pop[order[0]]
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_coeffs = tuple(gen_best.tolist())
        history.append(best_fit)
    return best_coeffs, best_fit, history

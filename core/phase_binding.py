"""
Phase-binding (binding by synchrony, Singer & Gray) с нуля — Kuramoto-style
связанные осцилляторы (веб-поиск 2026-09-01: Kuramoto dynamics — стандартная
математическая модель для этой гипотезы в литературе, не цитируем, а
реализуем). Каждая живая клетка получает фазу theta. Связь между соседями
взвешена ЛОКАЛЬНЫМ сходством химии (косинусное сходство chemistry-каналов) -
похожие по представлению соседи синхронизируются по фазе, непохожие - нет.
Никакого обучения весов, чистая динамика по явной формуле, как и в
vsa_binding.py.

dθ_i/dt = K * Σ_{j∈neighbors(i)} sim(i,j) * sin(θ_j - θ_i)

Гипотеза (the binding problem, Singer/Gray): если два пространственно
раздельных, химически различных "объекта" присутствуют ОДНОВРЕМЕННО, фаза
должна САМООРГАНИЗОВАННО синхронизироваться ВНУТРИ каждого объекта и
десинхронизироваться МЕЖДУ ними — без единого глобального управления,
только из локальных попарных взаимодействий.
"""
import torch
import torch.nn.functional as F

NEIGHBOR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def kuramoto_step(chem, phase, alive, K=1.0, dt=0.5, sim_gate=True):
    """chem: (C,H,W) химические каналы. phase: (H,W) в радианах. alive: (H,W) bool.
    sim_gate=False - контрольная группа: связь по чистой топологии (все соседи
    равновесно связаны), без взвешивания по сходству - baseline "просто соседи
    синхронизируются от близости", не от сходства представления.

    ВАЖНО (найдено во время отладки - см. VERIFICATION_LOG): clamp(min=0) на
    сходстве НЕ даёт сегментации в принципе - Kuramoto с НЕОТРИЦАТЕЛЬНОЙ связью
    на связном графе математически ВСЕГДА сходится к одной глобальной фазе
    (это известный факт теории Курамото, не баг реализации). Для реальной
    десинхронизации непохожих соседей связь должна быть ОТРИЦАТЕЛЬНОЙ
    (отталкивающей) при низком/отрицательном сходстве - не просто "слабой"."""
    chem_n = F.normalize(chem, dim=0)
    alive_f = alive.float()
    dtheta = torch.zeros_like(phase)
    for dy, dx in NEIGHBOR_OFFSETS:
        chem_shift = torch.roll(chem_n, shifts=(dy, dx), dims=(1, 2))
        theta_shift = torch.roll(phase, shifts=(dy, dx), dims=(0, 1))
        alive_shift = torch.roll(alive_f, shifts=(dy, dx), dims=(0, 1))
        if sim_gate:
            sim = (chem_n * chem_shift).sum(dim=0)  # знаковое сходство - НЕ clamp(min=0)
        else:
            sim = torch.ones_like(phase)
        coupling = sim * alive_shift * alive_f
        dtheta = dtheta + coupling * torch.sin(theta_shift - phase)
    new_phase = phase + dt * K * dtheta
    return new_phase


def order_parameter(phase, mask):
    """Kuramoto order parameter r = |mean(exp(i*theta))| по заданной маске клеток -
    r=1 - идеальная синхронность, r≈0 - фазы случайны/не связаны."""
    sel = phase[mask]
    if sel.numel() == 0:
        return 0.0
    z = torch.complex(torch.cos(sel), torch.sin(sel)).mean()
    return float(z.abs().item())

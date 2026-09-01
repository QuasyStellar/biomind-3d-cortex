import sys, time
sys.path.append("/content")
import torch
from core.unified_organism import LivingTissue

torch.manual_seed(42)
DEV = "cuda"


def organism_to_cuda(o):
    o.state = o.state.to(DEV)
    o.stress_ema = o.stress_ema.to(DEV)
    o.growth_ema = o.growth_ema.to(DEV)
    o.inflammation = o.inflammation.to(DEV)
    o.ctx_kernels = o.ctx_kernels.to(DEV)
    o.dg_proj = o.dg_proj.to(DEV)
    o.W_fast = o.W_fast.to(DEV)
    g = o.genome
    g.W = [w.to(DEV) for w in g.W]
    g.b = [b.to(DEV) for b in g.b]
    if g.adam:
        g.mW = [m.to(DEV) for m in g.mW]
        g.vW = [v.to(DEV) for v in g.vW]
        g.mb = [m.to(DEV) for m in g.mb]
        g.vb = [v.to(DEV) for v in g.vb]
    if o._replay_ctx is not None:
        o._replay_ctx = o._replay_ctx.to(DEV)
        o._replay_target = o._replay_target.to(DEV)
    return o


def signal(t, size):
    s = torch.zeros(1, 2, size, size, device=DEV)
    c = size // 2
    r = size // 4
    s[0, 0, c - r:c + r, c - r:c + r] = 0.5
    return s


def run():
    SIZE = 128
    organism = LivingTissue(size=SIZE, state_dim=16, seed=1)
    organism_to_cuda(organism)

    print("=" * 70)
    print("Рост до стабилизации (300 шагов)...")
    counts = []
    for t in range(300):
        n, err = organism.step(sensory_signal=signal(t, SIZE), train_genome=True)
        counts.append(n)
    print(f"Стабилизировалось: {counts[-1]} клеток")

    print("\n10 ЦИКЛОВ повреждение(30%) -> 250 шагов восстановления:")
    print("Вопрос: деградирует ли способность к регенерации, или популяция")
    print("сходится к natural carrying capacity (растёт, но overshoot% падает)?")
    cycle_results = []
    for cycle in range(1, 11):
        pre = int((organism.state[0, 0] > 0.1).sum().item())
        killed = organism.ablate(fraction=0.3)
        after_ablation = int((organism.state[0, 0] > 0.1).sum().item())

        t0 = time.time()
        for t in range(250):
            n, err = organism.step(sensory_signal=signal(t, SIZE), train_genome=True)
        recovered = n
        recovery_pct = 100.0 * recovered / pre if pre > 0 else 0.0
        cycle_results.append((cycle, pre, killed, after_ablation, recovered, recovery_pct))
        print(f"  Цикл {cycle}: было={pre:4d} убито={killed:4d} после_абляции={after_ablation:4d} "
              f"-> восстановлено={recovered:4d} ({recovery_pct:5.1f}%)  [{time.time()-t0:.1f}s]")

    print("\n" + "=" * 70)
    print("ИТОГ: устойчивость регенерации по циклам:")
    pcts = [r[5] for r in cycle_results]
    abs_pop = [r[4] for r in cycle_results]  # восстановленная популяция каждого цикла
    print(f"  Recovery % (относительно пред-абляционного) по циклам: {[round(p,1) for p in pcts]}")
    print(f"  Абсолютная популяция после восстановления по циклам: {abs_pop}")
    # Честная проверка: относительный % может падать ЕСТЕСТВЕННО, если популяция
    # сходится к carrying capacity среды (стимул фиксированного размера), а не
    # потому что способность к регенерации реально деградирует - смотрим на
    # АБСОЛЮТНЫЙ тренд, не только на процент.
    still_growing = all(abs_pop[i] < abs_pop[i+1] for i in range(len(abs_pop)-1))
    plateaued = abs(abs_pop[-1] - abs_pop[-2]) < 0.05 * abs_pop[-2] if len(abs_pop) > 1 else False
    if still_growing:
        verdict = "популяция монотонно растёт весь тест - НЕ деградация, скорее сходимость к carrying capacity (% падает естественно, т.к. база растёт)"
    elif plateaued:
        verdict = "популяция вышла на плато - похоже на настоящую carrying capacity, не деградацию"
    else:
        verdict = "популяция перестала расти или упала - требует более пристального взгляда, возможна настоящая деградация"
    print(f"  Честный вывод: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    run()

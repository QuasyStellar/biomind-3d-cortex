"""
M7 (docs/ROADMAP.md): "Структурная регенерация на систематическом масштабе".

Гипотеза: percentile-порог роста + локальное воспаление дают статистически
значимое (N>=8 seed, не единичный прогон) преимущество восстановления после
повреждения над baseline с ФИКСИРОВАННЫМ порогом роста (без percentile-
самонастройки, без воспаления) на масштабе 128x128, множественных severity
повреждения (15/30/50%), 10 последовательных циклов.

FIXED_THRESHOLD откалиброван отдельно (_m7_calibrate_fixed_threshold.py) как
медиана значения, к которому САМ adaptive-порог сходится при стабилизации на
этом масштабе - честный baseline (лучшее константное число, которое мог бы
выбрать исследователь без самонастройки), а не соломенное чучело.

Метрики за цикл: % восстановления популяции (относительно пред-абляционной)
и recall памяти (W_fast, синаптическая метка) - память пишется ОДИН раз
после стабилизации, проверяется после КАЖДОГО цикла восстановления структуры
(ожидание: не зависит от структурного повреждения, W_fast - отдельная
структура, но проверяем явно на масштабе, не предполагаем).
"""
import sys, time, json
sys.path.append("/content/repo")
import torch
import torch.nn.functional as F
from core.unified_organism import LivingTissue

DEV = "cuda"
FIXED_THRESHOLD = 1.36
SIZE = 128
N_FACTS = 40


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


def run_one(seed, condition, severity):
    torch.manual_seed(seed)
    kwargs = dict(size=SIZE, state_dim=16, seed=seed)
    if condition == "fixed":
        kwargs["growth_fixed_threshold"] = FIXED_THRESHOLD
        kwargs["inflammation_enabled"] = False
    organism = LivingTissue(**kwargs)
    organism_to_cuda(organism)

    for t in range(300):
        organism.step(sensory_signal=signal(t, SIZE), train_genome=True)

    g = torch.Generator().manual_seed(seed * 1000 + 7)
    fast_dim = organism.fast_dim
    keys = F.normalize(torch.randn(N_FACTS, fast_dim, generator=g), dim=-1).to(DEV)
    values = F.normalize(torch.randn(N_FACTS, fast_dim, generator=g), dim=-1).to(DEV)
    for i in range(N_FACTS):
        organism.write_fact(keys[i], values[i], tag_strength=1.0)

    def recall_score():
        correct = 0
        for i in range(N_FACTS):
            pred = organism.read_fact(keys[i])
            sims = F.cosine_similarity(values, pred.unsqueeze(0), dim=-1)
            correct += int(sims.argmax().item() == i)
        return correct

    rows = []
    for cycle in range(1, 11):
        pre = int((organism.state[0, 0] > 0.1).sum().item())
        killed = organism.ablate(fraction=severity)
        n = pre
        for t in range(250):
            n, err = organism.step(sensory_signal=signal(t, SIZE), train_genome=True)
        recovery_pct = 100.0 * n / pre if pre > 0 else 0.0
        rows.append(dict(cycle=cycle, pre=pre, killed=killed, recovered=n,
                          recovery_pct=recovery_pct, recall=recall_score()))
    return rows


def main():
    SEEDS = list(range(1, 9))
    CONDITIONS = ["adaptive", "fixed"]
    SEVERITIES = [0.15, 0.30, 0.50]
    all_rows = []
    t0 = time.time()
    for condition in CONDITIONS:
        for severity in SEVERITIES:
            for seed in SEEDS:
                for r in run_one(seed, condition, severity):
                    row = dict(condition=condition, severity=severity, seed=seed, **r)
                    all_rows.append(row)
                print(f"# done seed={seed} condition={condition} severity={severity} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
    with open("/content/m7_results.json", "w") as f:
        json.dump(all_rows, f)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

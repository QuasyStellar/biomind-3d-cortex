"""M7: calibrate a single constant growth threshold for the fixed-threshold
baseline at size=128, by observing what value the percentile-adaptive
threshold itself converges to during stabilization. This gives the fixed
baseline the best single number a researcher could plausibly have picked
without percentile self-tuning - not a strawman."""
import sys
sys.path.append("/content/repo")
import torch
from core.unified_organism import LivingTissue


def signal(t, size):
    s = torch.zeros(1, 2, size, size)
    c = size // 2
    r = size // 4
    s[0, 0, c - r:c + r, c - r:c + r] = 0.5
    return s


torch.manual_seed(42)
o = LivingTissue(size=128, state_dim=16, seed=1)
thresholds = []
for t in range(300):
    n, err = o.step(sensory_signal=signal(t, 128), train_genome=True)
    if o._step_count % o.growth_period == 0:
        alive, _ = o.alive_mask()
        alive_growth_ema = o.growth_ema[alive]
        if alive_growth_ema.numel() >= 4:
            th = torch.quantile(alive_growth_ema, o.growth_percentile / 100.0).item()
            thresholds.append(th)
print("last 10 thresholds:", [round(x, 4) for x in thresholds[-10:]])
print("median of last 10:", sorted(thresholds[-10:])[5])
print("final cell count:", n)

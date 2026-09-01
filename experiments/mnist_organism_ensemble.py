"""
Рычаг 3 ("что нужно для 80%"): честный ансамбль из НЕСКОЛЬКИХ независимо
выращенных тканей (разные seed - разная история роста/bootstrap/PC-
релаксации генома), каждая даёт свой linear-probe recall на ОДНИХ и тех
же test-цифрах, голоса усредняются. Не путать с провалившейся попыткой
"ансамбль ОДНОЙ ткани двумя методами чтения" (VERIFICATION_LOG) - там
один из двух методов не сошёлся, ансамбль был испорчен слабым участником.
Здесь ВСЕ участники - одинаково устроенные, независимо честно обученные
linear-probe системы (тот же протокол, что дал 68.3%/69.0%), различается
только seed.
"""
import sys, os, time, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from core.unified_organism import LivingTissue
from core.mnist_loader import load_mnist
import experiments.mnist_organism_classification as base

N_SEEDS = 5
N_TRAIN = 600
N_TEST = 500


def train_one_tissue(seed, tr_x, tr_y, train_idx):
    torch.manual_seed(seed)
    organism = LivingTissue(size=base.SIZE, state_dim=16, seed=seed, genome_hidden=base.GENOME_HIDDEN)
    for t in range(base.GROWTH_STEPS):
        n, err = organism.step(sensory_signal=base.blob_signal(t), train_genome=True)
    for cycle in range(base.BOOTSTRAP_CYCLES):
        organism.ablate(fraction=base.BOOTSTRAP_FRACTION)
        for t in range(base.BOOTSTRAP_STEPS):
            n, err = organism.step(sensory_signal=base.blob_signal(t), train_genome=True)
    organism.growth_enabled = False
    snapshot = copy.deepcopy(organism)

    baseline = base.raw_hidden(copy.deepcopy(snapshot), torch.zeros(base.DIGIT_SIDE, base.DIGIT_SIDE),
                                base.K_STEPS_PER_DIGIT, pool=base.SPATIAL_POOL)
    organism.add_modality("mnist_readout", key_dim=baseline.numel(), value_dim=10, seed=99,
                           sdr_dim=base.SDR_DIM, sparsity=base.SDR_SPARSITY)

    for i in train_idx:
        image, label = tr_x[i], tr_y[i].item()
        h = base.raw_hidden(copy.deepcopy(snapshot), image, base.K_STEPS_PER_DIGIT, pool=base.SPATIAL_POOL)
        feat = F.normalize(h - baseline, dim=0)
        onehot = torch.zeros(10)
        onehot[label] = 1.0
        organism.write_fact_modal("mnist_readout", feat, onehot, tag_strength=1.0)

    return organism, snapshot, baseline


def predict_one_tissue(organism, snapshot, baseline, image):
    h = base.raw_hidden(copy.deepcopy(snapshot), image, base.K_STEPS_PER_DIGIT, pool=base.SPATIAL_POOL)
    feat = F.normalize(h - baseline, dim=0)
    return organism.read_fact_modal("mnist_readout", feat)


def run():
    tr_x, tr_y, te_x, te_y = load_mnist()
    g = torch.Generator().manual_seed(100)
    train_idx = torch.randperm(tr_x.shape[0], generator=g)[:N_TRAIN].tolist()
    test_idx = torch.randperm(te_x.shape[0], generator=torch.Generator().manual_seed(200))[:N_TEST].tolist()

    members = []
    t0 = time.time()
    for seed in range(1, N_SEEDS + 1):
        organism, snapshot, baseline = train_one_tissue(seed, tr_x, tr_y, train_idx)
        members.append((organism, snapshot, baseline))
        print(f"tissue seed={seed} trained ({time.time()-t0:.1f}s)")

    per_member_correct = [0] * N_SEEDS
    ensemble_correct = 0
    per_class_correct = torch.zeros(10)
    per_class_total = torch.zeros(10)
    t0 = time.time()
    for count, i in enumerate(test_idx):
        image, label = te_x[i], te_y[i].item()
        votes_sum = torch.zeros(10)
        for m_idx, (organism, snapshot, baseline) in enumerate(members):
            pred = predict_one_tissue(organism, snapshot, baseline, image)
            per_member_correct[m_idx] += int(pred.argmax().item() == label)
            votes_sum += F.normalize(pred, dim=0)
        ens_label = int(votes_sum.argmax().item())
        ensemble_correct += int(ens_label == label)
        per_class_total[label] += 1
        per_class_correct[label] += int(ens_label == label)
        if (count + 1) % 100 == 0:
            print(f"  tested {count+1}/{N_TEST} ({time.time()-t0:.1f}s)")

    print("=" * 70)
    for m_idx in range(N_SEEDS):
        print(f"tissue seed={m_idx+1}: {per_member_correct[m_idx]/N_TEST*100:.1f}%")
    print(f"ENSEMBLE ({N_SEEDS} tissues, N_test={N_TEST}): {ensemble_correct/N_TEST*100:.1f}%")
    print("По классам (ансамбль):")
    for c in range(10):
        tot = int(per_class_total[c].item())
        cor = int(per_class_correct[c].item())
        pct = 100.0 * cor / tot if tot > 0 else float("nan")
        print(f"  {c}: {cor}/{tot} ({pct:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run()

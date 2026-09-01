import sys, time
sys.path.append("/content")
import torch
torch.set_default_device("cuda")

# core/*.py создаёт torch.Generator() без device (валидировано и работает
# на CPU локально) - под глобальным cuda-дефолтом это даёт device mismatch.
# Не трогаем провалидированный код - патчим только здесь, в колаб-драйвере.
_orig_generator = torch.Generator
def _cuda_generator(*args, **kwargs):
    kwargs.setdefault("device", "cuda")
    return _orig_generator(*args, **kwargs)
torch.Generator = _cuda_generator

from core.unified_organism import LivingTissue

torch.manual_seed(42)

SIZE = 128  # 16384 max cells vs 24x24=576 locally - ~28x larger canvas
STATE_DIM = 16


def signal(t, size):
    s = torch.zeros(1, 2, size, size)
    c = size // 2
    r = size // 4
    s[0, 0, c - r:c + r, c - r:c + r] = 0.5
    return s


def run():
    t0 = time.time()
    organism = LivingTissue(size=SIZE, state_dim=STATE_DIM, seed=1)
    print(f"Init done in {time.time()-t0:.1f}s")

    print("=" * 70)
    print(f"1) Рост на масштабе {SIZE}x{SIZE} (макс {SIZE*SIZE} клеток), {STATE_DIM}-dim state:")
    t0 = time.time()
    STEPS = 1500
    counts, errors = [], []
    for t in range(STEPS):
        n, err = organism.step(sensory_signal=signal(t, SIZE), train_genome=True)
        counts.append(n)
        errors.append(err)
        if t % 150 == 0 or t == STEPS - 1:
            print(f"   шаг={t:4d}  живых клеток={n:6d}  ошибка={err:.4f}  "
                  f"({time.time()-t0:.1f}s elapsed)")
    print(f"   Клеток: {counts[0]} -> {counts[-1]}")
    print(f"   Ошибка: {errors[10]:.4f} (ранняя) -> {sum(errors[-50:])/50:.4f} (поздняя, среднее по 50)")
    print(f"   Время на {STEPS} шагов: {time.time()-t0:.1f}s ({(time.time()-t0)/STEPS*1000:.1f}ms/шаг)")

    print("\n2) Быстрая память с приоритетом по метке, N=400 фактов (10x больше локального теста):")
    g = torch.Generator(device="cuda").manual_seed(7)
    fast_dim = organism.fast_dim
    n_facts = 400
    keys = torch.nn.functional.normalize(torch.randn(n_facts, fast_dim, generator=g), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(n_facts, fast_dim, generator=g), dim=-1)
    tags = torch.tensor([1.5 if i % 2 == 0 else 0.3 for i in range(n_facts)])
    t0 = time.time()
    for i in range(n_facts):
        organism.write_fact(keys[i], values[i], tag_strength=tags[i].item())
    print(f"   Запись {n_facts} фактов: {time.time()-t0:.1f}s")

    def decode(vec):
        sims = torch.nn.functional.cosine_similarity(values, vec.unsqueeze(0), dim=-1)
        return int(sims.argmax().item())

    high_correct = low_correct = 0
    for i in range(n_facts):
        pred = organism.read_fact(keys[i])
        correct = (decode(pred) == i)
        if tags[i] > 1.0:
            high_correct += correct
        else:
            low_correct += correct
    print(f"   Высокая метка: {high_correct}/{n_facts//2} ({100*high_correct/(n_facts//2):.1f}%)")
    print(f"   Низкая метка:  {low_correct}/{n_facts//2} ({100*low_correct/(n_facts//2):.1f}%)")

    print("\n3) Повреждение + воспаление-регенерация на масштабе:")
    pre_count = counts[-1]
    killed = organism.ablate(fraction=0.4)
    alive_after = int((organism.state[0, 0] > 0.1).sum().item())
    print(f"   Убито: {killed} (было {pre_count}, осталось {alive_after})")

    recall_post_damage = sum(decode(organism.read_fact(keys[i])) == i for i in range(n_facts))
    print(f"   Recall сразу после повреждения: {recall_post_damage}/{n_facts}")

    t0 = time.time()
    REC_STEPS = 600
    rec_counts = []
    for t in range(REC_STEPS):
        n, err = organism.step(sensory_signal=signal(t, SIZE), train_genome=True)
        rec_counts.append(n)
        if t % 100 == 0 or t == REC_STEPS - 1:
            infl = organism.inflammation.sum().item()
            print(f"   восст. шаг={t:4d}  n={n:6d}  inflammation_sum={infl:.2f}")
    final_n = rec_counts[-1]
    print(f"   Восстановление: {alive_after} -> {final_n} клеток "
          f"({100*final_n/pre_count:.1f}% от до-абляционного), {time.time()-t0:.1f}s")

    recall_final = sum(decode(organism.read_fact(keys[i])) == i for i in range(n_facts))
    print(f"   Recall после восстановления: {recall_final}/{n_facts}")
    print("=" * 70)


if __name__ == "__main__":
    run()

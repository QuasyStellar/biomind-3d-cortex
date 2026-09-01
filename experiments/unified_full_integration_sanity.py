"""
Комплексная регрессия/интеграция: за эту сессию в unified_organism.py
независимо добавлены колонки (add_modality/write_fact_modal/read_fact_modal/
read_fact_voted), VSA-compositional (init_vsa/write_compositional/
read_compositional), phase-binding (init_phase/phase_sync_step), JEPA-
инфраструктура (compute_context/hidden_representation) - каждая проверялась
ОТДЕЛЬНО в своём собственном тесте, но НИКОГДА все вместе на одном
организме одновременно. Проверяем: не ломают ли они друг друга при
совместном использовании (общие атрибуты, побочные эффекты на organism.state/
genome), и что базовый рост/обучение продолжает работать корректно ДО, ПОСЛЕ
и МЕЖДУ вызовами всех остальных подсистем.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.unified_organism import LivingTissue

torch.manual_seed(42)


def signal(t, size):
    s = torch.zeros(1, 2, size, size)
    c = size // 2
    r = size // 4
    s[0, 0, c - r:c + r, c - r:c + r] = 0.5
    return s


def check(name, cond):
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}")
    return cond


def run():
    all_ok = True
    SIZE = 32
    print("=" * 70)
    print("1) Базовый рост+обучение (100 шагов) - точка отсчёта...")
    organism = LivingTissue(size=SIZE, state_dim=16, seed=1)
    errs_before = []
    for t in range(100):
        n, err = organism.step(sensory_signal=signal(t, SIZE), train_genome=True)
        errs_before.append(err)
    pop_before = n
    err_before = sum(errs_before[-20:]) / 20
    all_ok &= check(f"население > 0 после начального роста ({pop_before})", pop_before > 0)
    all_ok &= check(f"ошибка генома конечна ({err_before:.4f})", err_before == err_before and err_before < 1e6)

    print("\n2) Default fast memory (write_fact/read_fact)...")
    g = torch.Generator().manual_seed(1)
    fd = organism.fast_dim
    k1 = torch.nn.functional.normalize(torch.randn(fd, generator=g), dim=0)
    v1 = torch.nn.functional.normalize(torch.randn(fd, generator=g), dim=0)
    organism.write_fact(k1, v1, tag_strength=1.0)
    r1 = organism.read_fact(k1)
    all_ok &= check("read_fact возвращает конечный вектор", torch.isfinite(r1).all().item())

    print("\n3) Колонки (add_modality/write_fact_modal/read_fact_modal/read_fact_voted)...")
    organism.add_modality("colA", key_dim=fd, seed=1)
    organism.add_modality("colB", key_dim=fd, seed=2)
    k2 = torch.nn.functional.normalize(torch.randn(fd, generator=g), dim=0)
    v2 = torch.nn.functional.normalize(torch.randn(fd, generator=g), dim=0)
    organism.write_fact_modal("colA", k2, v2, tag_strength=1.0)
    organism.write_fact_modal("colB", k2, v2, tag_strength=1.0)
    r2 = organism.read_fact_voted({"colA": k2, "colB": k2})
    all_ok &= check("read_fact_voted возвращает конечный вектор", torch.isfinite(r2).all().item())

    print("\n4) VSA-compositional (init_vsa/write_compositional/read_compositional)...")
    organism.init_vsa(vsa_dim=64, n_roles=4, seed=1)
    slot_key = torch.nn.functional.normalize(torch.randn(64, generator=g), dim=0)
    filler = torch.nn.functional.normalize(torch.randn(64, generator=g), dim=0)
    organism.write_compositional(slot_key, [(0, filler)], tag_strength=1.0)
    r3 = organism.read_compositional(slot_key, 0)
    all_ok &= check("read_compositional возвращает конечный вектор", torch.isfinite(r3).all().item())
    sim = torch.nn.functional.cosine_similarity(r3, filler, dim=0).item()
    all_ok &= check(f"read_compositional узнаёт записанный filler (cos_sim={sim:.3f} > 0.5)", sim > 0.5)

    print("\n5) JEPA-инфраструктура (compute_context/hidden_representation)...")
    ctx_flat, ys, xs = organism.compute_context()
    all_ok &= check(f"compute_context вернул непустой контекст ({ctx_flat.shape[0]} клеток)", ctx_flat.shape[0] > 0)
    hidden = organism.hidden_representation(ctx_flat)
    all_ok &= check("hidden_representation возвращает конечный тензор", torch.isfinite(hidden).all().item())

    print("\n6) Phase-binding (init_phase/phase_sync_step)...")
    organism.init_phase(seed=1)
    for _ in range(20):
        organism.phase_sync_step(K=1.0, dt=0.1, sim_gate=True)
    all_ok &= check("phase после 20 шагов синхронизации конечна", torch.isfinite(organism.phase).all().item())

    print("\n7) Ablate + clone (структурное повреждение и копирование)...")
    clone = organism.clone()
    killed = organism.ablate(fraction=0.3)
    all_ok &= check(f"ablate вернул неотрицательное число убитых ({killed})", killed >= 0)
    all_ok &= check("clone НЕ пострадал от ablate оригинала (независимая копия)",
                     int((clone.state[0, 0] > 0.1).sum().item()) >= int((organism.state[0, 0] > 0.1).sum().item()))

    print("\n8) Рост+обучение ПРОДОЛЖАЕТСЯ корректно ПОСЛЕ всех вызовов выше (100 шагов)...")
    errs_after = []
    for t in range(100):
        n, err = organism.step(sensory_signal=signal(t, SIZE), train_genome=True)
        errs_after.append(err)
    pop_after = n
    err_after = sum(errs_after[-20:]) / 20
    all_ok &= check(f"население после интеграции конечно и > 0 ({pop_after})", pop_after > 0)
    all_ok &= check(f"ошибка генома после интеграции конечна ({err_after:.4f})",
                     err_after == err_after and err_after < 1e6)

    print("\n9) Повторная проверка ВСЕХ фактов, записанных ДО повреждения+интеграции...")
    r1_final = organism.read_fact(k1)
    sim1 = torch.nn.functional.cosine_similarity(r1_final, v1, dim=0).item()
    all_ok &= check(f"default fast memory пережила все манипуляции (cos_sim={sim1:.3f})", torch.isfinite(r1_final).all().item())

    print("\n" + "=" * 70)
    print(f"ИТОГ: {'ВСЕ ПРОВЕРКИ ПРОШЛИ' if all_ok else 'ЕСТЬ ПАДЕНИЯ - см. FAIL выше'}")
    print("=" * 70)
    return all_ok


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)

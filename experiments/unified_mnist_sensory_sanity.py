"""
Продолжение приоритета (d): реальные сенсорные данные подключены к самому
организму (unified_organism.py), не только к изолированному PC-net сравнению
(mnist_pc_vs_backprop_sanity.py). Настоящая MNIST-цифра используется как
персистентный сенсорный стимул, управляющий ростом ткани - вместо синтетического
однородного квадратного пятна, которое использовалось во ВСЕХ прошлых тестах
unified_organism.py.

Вопрос: self-supervised геном (предсказывает identity клетки по контексту
соседей - тот же принцип, что и JEPA) учится ли осмысленной ПРОСТРАНСТВЕННОЙ
структуре реального изображения (края штриха, скоррелированные соседние
пиксели) лучше, чем на случайном шуме той же интенсивности/размера (нет
пространственной структуры вообще)? Честный baseline - шум с ТЕМИ ЖЕ
статистиками (среднее/std), не "цифра vs пусто".
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.unified_organism import LivingTissue
from core.mnist_loader import load_mnist

torch.manual_seed(42)

SIZE = 28  # естественный размер MNIST - не ресайзим, тестируем как есть


def digit_signal(t, image):
    """image: (28,28) в [0,1] - персистентный сенсорный вход, постоянный по t
    (то же самое изображение управляет ростом каждый шаг, как и синтетический
    квадрат в остальных тестах - персистентный, не мигающий, стимул)."""
    s = torch.zeros(1, 2, SIZE, SIZE)
    s[0, 0] = image * 0.5
    return s


def run(seed=1, steps=400):
    tr_x, tr_y, te_x, te_y = load_mnist()
    g = torch.Generator().manual_seed(seed)
    digit_idx = torch.randint(0, tr_x.shape[0], (1,), generator=g).item()
    digit_image = tr_x[digit_idx]
    digit_label = tr_y[digit_idx].item()
    print(f"Цифра: label={digit_label}, индекс={digit_idx}")

    # ИСПРАВЛЕНО (найден баг): normalize-to-mean/std + clamp(0,1) искажает
    # статистики ПОСЛЕ clamp (пиксели цифры сильно бимодальны - в основном
    # чистый фон + яркий штрих, std после клипа уже не совпадает с целевым) -
    # печать показала digit std=0.259 vs "смэтченный" noise std=0.181, то есть
    # никакого честного совпадения не было. Вместо этого - shuffle тех же
    # САМЫХ пикселей цифры по случайным позициям: гарантированно ТОЧНО те же
    # mean/std/min/max/гистограмма, разрушена только пространственная
    # структура (соседство), что и является единственной проверяемой переменной.
    noise_image = digit_image.flatten()[torch.randperm(SIZE * SIZE, generator=g)].reshape(SIZE, SIZE)
    print(f"digit: mean={digit_image.mean():.3f} std={digit_image.std():.3f} | "
          f"noise (shuffled pixels of SAME digit): mean={noise_image.mean():.3f} std={noise_image.std():.3f}")

    def run_condition(image, seed_offset):
        organism = LivingTissue(size=SIZE, state_dim=16, seed=seed + seed_offset)
        errors, counts = [], []
        for t in range(steps):
            n, err = organism.step(sensory_signal=digit_signal(t, image), train_genome=True)
            errors.append(err)
            counts.append(n)
        return errors, counts

    print("\nРост под реальной MNIST-цифрой...")
    err_digit, cnt_digit = run_condition(digit_image, 0)
    print(f"   финал: {cnt_digit[-1]} клеток, ошибка генома (посл. 50 шагов) = {sum(err_digit[-50:])/50:.4f}")

    print("Рост под шумом (тот же mean/std, без пространственной структуры)...")
    err_noise, cnt_noise = run_condition(noise_image, 0)
    print(f"   финал: {cnt_noise[-1]} клеток, ошибка генома (посл. 50 шагов) = {sum(err_noise[-50:])/50:.4f}")

    print("=" * 70)
    e_digit, e_noise = sum(err_digit[-50:]) / 50, sum(err_noise[-50:]) / 50
    print(f"Ошибка генома (ниже = лучше научился): digit={e_digit:.4f}  noise={e_noise:.4f}")
    if e_digit < e_noise * 0.9:
        print("=> Геном использует реальную пространственную структуру цифры лучше, чем шум")
    elif e_digit > e_noise * 1.1:
        print("=> Неожиданно: на цифре ошибка ВЫШЕ, чем на шуме - зафиксировано честно, требует объяснения")
    else:
        print("=> Разница незначительна на этом масштабе/N=1 цифра - не подтверждено")
    print("=" * 70)

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(digit_image, cmap="gray")
    axes[0].set_title(f"MNIST digit (label={digit_label})")
    axes[1].plot(err_digit, label="digit", color="darkorange")
    axes[1].plot(err_noise, label="noise (matched mean/std)", color="steelblue")
    axes[1].set_title("Ошибка предсказания генома")
    axes[1].set_xlabel("Шаг")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(cnt_digit, label="digit", color="darkorange")
    axes[2].plot(cnt_noise, label="noise", color="steelblue")
    axes[2].set_title("Популяция")
    axes[2].set_xlabel("Шаг")
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    path = os.path.join(plots_dir, "unified_mnist_sensory_sanity.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved: {path}")

    return e_digit, e_noise


if __name__ == "__main__":
    run()

"""
M9: структурированность реальных пикселей MNIST по ТОЙ ЖЕ формуле, что
`m9_structuredness_vs_gap.py` использует для embedding-векторов (средняя
корреляция Пирсона между соседними измерениями входного вектора) - чтобы
поставить уже готовый результат PC-vs-BP разрыва на MNIST (~1.79 п.п.,
9 прогонов, см. VERIFICATION_LOG) на ОДНУ ось со sigma-развёрткой
синтетических embeddings, не пересчитывая сам PC/BP-тест заново.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.mnist_loader import load_mnist


def structuredness(X):
    Xz = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-7)
    prod = (Xz[:, :-1] * Xz[:, 1:]).mean(dim=0)
    return prod.mean().item()


if __name__ == "__main__":
    train_images, _, _, _ = load_mnist(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mnist"))
    flat = train_images[:5000].reshape(5000, -1)  # (5000, 784), raster-order пиксели
    s = structuredness(flat)
    print(f"MNIST (N=5000, 784 сырых пикселя, raster-порядок): structuredness={s:+.4f}")
    print("Известный PC-vs-BP разрыв на MNIST (9 прогонов, N_train in {1000,2000,4000}x3 seed): "
          "+1.79+-0.75 п.п. в пользу BP (см. VERIFICATION_LOG)")

"""
VSA / HRR (Holographic Reduced Representations, Plate 1995) с нуля:
связывание через circular convolution (в Фурье-домене — поэлементное
произведение), развязывание — через корреляцию с приближённым обратным
(инволюция вектора). Никакого обучения весов — чистая алгебра над
случайными векторами. "The Binding Problem" здесь — вопрос ёмкости:
сколько одновременно связанных пар можно суперпозировать (сложить) в
один вектор и всё ещё корректно развязать обратно.
"""
import torch


def circular_conv(a, b):
    """Связывание: a ⊛ b через FFT (поэлементное произведение в частотном домене)."""
    fa = torch.fft.fft(a)
    fb = torch.fft.fft(b)
    return torch.fft.ifft(fa * fb).real


def involution(a):
    """Приближённый обратный элемент для развязывания: a^[0]=a[0], a^[i]=a[n-i]."""
    return torch.cat([a[:1], a.flip(0)[:-1]])


def circular_corr(c, a):
    """Развязывание: c ⊛ involution(a) ≈ b, если c = a ⊛ b."""
    return circular_conv(c, involution(a))


def random_vectors(n, dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, dim, generator=g) / (dim ** 0.5)
    return v

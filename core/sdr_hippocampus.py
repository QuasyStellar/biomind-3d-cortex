"""
SDR-ассоциативная память с нуля: Хеббовская запись без backprop, разреженный
код (kWTA) против интерференции. Плюс плотный (не разреженный) baseline —
чтобы честно измерить, действительно ли разреженность причина устойчивости,
а не что-то ещё.
"""
import torch


class SDRHippocampus:
    def __init__(self, dim, sdr_dim=1024, sparsity=0.06, beta=0.9, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.dim = dim
        self.sdr_dim = sdr_dim
        self.k = max(1, int(sdr_dim * sparsity))
        self.beta = beta
        self.dg_proj = torch.randn(sdr_dim, dim, generator=g) * (1.0 / dim ** 0.5)
        self.W = torch.zeros(dim, sdr_dim)

    def code(self, key):
        h = torch.relu(self.dg_proj @ key)
        val, idx = torch.topk(h, self.k)
        sdr = torch.zeros_like(h)
        sdr[idx] = val
        norm = sdr.norm() + 1e-7
        return sdr / norm

    def write(self, key, value):
        s = self.code(key)
        pred = self.W @ s
        err = value - pred
        self.W += self.beta * torch.outer(err, s)

    def read(self, key):
        s = self.code(key)
        return self.W @ s


class DenseHippocampus:
    """Тот же Hebbian-механизм, но без разреженного кода — прямой плотный ключ."""
    def __init__(self, dim, beta=0.9, seed=0):
        self.dim = dim
        self.beta = beta
        self.W = torch.zeros(dim, dim)

    def write(self, key, value):
        pred = self.W @ key
        err = value - pred
        self.W += self.beta * torch.outer(err, key)

    def read(self, key):
        return self.W @ key

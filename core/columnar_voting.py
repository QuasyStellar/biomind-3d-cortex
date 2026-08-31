"""
Колонки + голосование (Thousand Brains) вместо ручной формулы слияния
модальностей. Каждая колонка — независимая ассоциативная память (переиспользуем
уже проверенный SDRHippocampus) на СВОЙ модальный поток. Консенсус — простое
голосование (сумма нормированных косинусных сходств кандидатов от всех
активных колонок), без weight-per-modality констант, которые ломаются
при добавлении новой модальности.
"""
import torch
from core.sdr_hippocampus import SDRHippocampus


class Column:
    def __init__(self, dim, sdr_dim=512, sparsity=0.06, beta=0.9, seed=0):
        self.memory = SDRHippocampus(dim=dim, sdr_dim=sdr_dim, sparsity=sparsity, beta=beta, seed=seed)

    def learn(self, cue, target):
        self.memory.write(cue, target)

    def vote(self, cue, candidates):
        """Возвращает нормированные косинусные сходства этой колонки к каждому
        кандидату в общем пространстве понятий - это и есть "голос" колонки."""
        pred = self.memory.read(cue)
        if pred.norm() < 1e-6:
            return torch.zeros(candidates.shape[0])
        return torch.nn.functional.cosine_similarity(candidates, pred.unsqueeze(0), dim=-1)


def vote_consensus(columns, cues_by_modality, candidates):
    """Простое голосование: суммируем нормированные голоса всех АКТИВНЫХ
    (переданных в cues_by_modality) колонок - нет ручных весов на модальность,
    поэтому схема одинаково работает при 1, 2, 3 или 4 одновременных входах."""
    total = torch.zeros(candidates.shape[0])
    for m, cue in cues_by_modality.items():
        total = total + columns[m].vote(cue, candidates)
    return int(total.argmax().item())

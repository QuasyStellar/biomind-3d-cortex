"""
Temporal PC (`core/temporal_pc.py`, M10, Millidge et al. 2024 - статья
прочитана напрямую) на РЕАЛЬНОЙ последовательности (tinyshakespeare), не
игрушечном 2D-вращении (`temporal_pc_tracking_sanity.py`) - honest шаг
дальше, раз MNIST принципиально не может проверить временную часть плана
(статична, нет понятия "следующий момент").

Протокол: каждый символ кодируется ФИКСИРОВАННЫМ случайным embedding'ом
(как везде в M(-1)-семействе тестов - embedding не обучается, изолирует
"учится ли локальное правило чему-то полезному о ПОСЛЕДОВАТЕЛЬНОСТИ", не
об embedding-пространстве). На каждом шаге t: ДО того, как сеть увидит
y_t (embedding символа t), берём предсказание C @ f(x_hat_{t-1}) -
честный one-step-ahead прогноз СЛЕДУЮЩЕГО символа по всей предыдущей
истории - декодируем через ближайший (по косинусу) embedding в словаре
(тот же приём decode(), что и в char-LM тестах M(-1)). Сравниваем
top-1 accuracy предсказания СЛЕДУЮЩЕГО символа: обучаемая версия vs
ЗАМОРОЖЕННАЯ (weight_lr=0, та же архитектура и релаксация, но без
обучения - изолирует эффект именно локального правила) vs chance
(1/65 ~ 1.5%).
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from core.temporal_pc import TemporalPredictiveCoding

CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "text", "corpus.txt")


def load_corpus():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return text, stoi, len(chars)


def run(T=3000, x_dim=64, y_dim=32, seed=1, nonlinear=True):
    text, stoi, vocab_size = load_corpus()
    g = torch.Generator().manual_seed(seed)
    E = F.normalize(torch.randn(vocab_size, y_dim, generator=g), dim=-1)

    ids = [stoi[c] for c in text[:T]]

    def decode(vec):
        sims = F.cosine_similarity(E, vec.unsqueeze(0), dim=-1)
        return int(sims.argmax().item())

    net_learn = TemporalPredictiveCoding(x_dim=x_dim, y_dim=y_dim, relax_steps=10, relax_dt=0.1,
                                          weight_lr=0.02, seed=seed, nonlinear=nonlinear)
    net_frozen = TemporalPredictiveCoding(x_dim=x_dim, y_dim=y_dim, relax_steps=10, relax_dt=0.1,
                                           weight_lr=0.0, seed=seed, nonlinear=nonlinear)

    correct_learn, correct_frozen = [], []
    t0 = time.time()
    for t in range(T):
        y_t = E[ids[t]]
        pred_learn = net_learn.predict_y()
        pred_frozen = net_frozen.predict_y()
        correct_learn.append(int(decode(pred_learn) == ids[t]))
        correct_frozen.append(int(decode(pred_frozen) == ids[t]))
        net_learn.step(y_t)
        net_frozen.step(y_t)
        if (t + 1) % 1000 == 0:
            print(f"  {t+1}/{T} ({time.time()-t0:.1f}s)")

    win = 200
    early_l = sum(correct_learn[:win]) / win
    late_l = sum(correct_learn[-win:]) / win
    early_f = sum(correct_frozen[:win]) / win
    late_f = sum(correct_frozen[-win:]) / win
    overall_l = sum(correct_learn) / T
    overall_f = sum(correct_frozen) / T
    print(f"vocab_size={vocab_size}  chance={1/vocab_size*100:.1f}%")
    print(f"learn : early={early_l*100:.1f}%  late={late_l*100:.1f}%  overall={overall_l*100:.1f}%")
    print(f"frozen: early={early_f*100:.1f}%  late={late_f*100:.1f}%  overall={overall_f*100:.1f}%")
    return early_l, late_l, overall_l, overall_f


if __name__ == "__main__":
    run()

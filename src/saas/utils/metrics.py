from typing import Iterable

def f1_score_framewise(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """
    F1 binário simples em escala 0..1.
    y_true/y_pred: iteráveis com 0/1.
    """
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1
    # precisão e recall com proteção a divisão por zero
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
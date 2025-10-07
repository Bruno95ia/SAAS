import pytest
from saas.utils import metrics

def test_f1_score_perfeito():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]
    score = metrics.f1_score_framewise(y_true, y_pred)
    print("DEBUG score perfeito:", score)  # -s para ver
    assert score == pytest.approx(1.0, rel=1e-6)

def test_f1_score_zero():
    y_true = [1, 1, 1]
    y_pred = [0, 0, 0]
    score = metrics.f1_score_framewise(y_true, y_pred)
    print("DEBUG score zero:", score)
    assert score == pytest.approx(0.0, rel=1e-6)
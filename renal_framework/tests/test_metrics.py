import numpy as np

from src.metrics import sigmoid, binary_metrics_from_logits, tune_threshold_for_f1_pos


def test_sigmoid_bounds():
    x = np.array([-10.0, 0.0, 10.0])
    y = sigmoid(x)
    assert np.all(y > 0.0)
    assert np.all(y < 1.0)
    assert np.isclose(y[1], 0.5, atol=1e-8)


def test_binary_metrics_from_logits_basic():
    y_true = np.array([0, 1, 0, 1])
    logits = np.array([-2.0, 2.0, -1.5, 1.5])

    out = binary_metrics_from_logits(y_true, logits, threshold=0.5)
    assert "f1_macro" in out and "f1_pos" in out
    assert out["f1_macro"] >= 0.0
    assert out["f1_pos"] >= 0.0
    assert out["probs"].shape == y_true.shape
    assert out["preds"].shape == y_true.shape


def test_tune_threshold_for_f1_pos_returns_valid():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    logits = np.array([-2.0, -1.0, 0.2, 0.8, 1.5, -0.3])

    thr, f1 = tune_threshold_for_f1_pos(y_true, logits, input_is_logits=True)
    assert 0.05 <= thr <= 0.95
    assert 0.0 <= f1 <= 1.0
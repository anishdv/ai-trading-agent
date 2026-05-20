import numpy as np
import pandas as pd

from src.ml.dataset import add_label, time_split
from src.ml.metrics import evaluate_probs


def test_label_no_leakage_on_features():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    close = pd.Series([100, 102, 101, 105, 104], index=dates)
    df = pd.DataFrame({"close": close})
    out = add_label(df)
    # last row dropped (no forward label)
    assert len(out) == 4
    assert out["label_up"].iloc[0] == 1  # 100 -> 102


def test_time_split_order():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame({"close": np.arange(100)}, index=dates)
    df["label_up"] = 0
    for c in ["ret_1d", "ret_3d", "rsi_14", "macd", "macd_signal", "macd_hist", "roll_std_20", "vol_ratio"]:
        df[c] = 0.0
    train, val, test = time_split(df)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_brier_perfect():
    y = np.array([1, 0, 1, 0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert evaluate_probs(y, p)["brier"] == 0.0

import numpy as np
import pandas as pd
import pytest
from src.data.features import build_feature_frame, returns, rsi_wilder, sma


def test_sma_and_returns():
    close = pd.Series([100.0, 110.0, 99.0])
    assert returns(close, 1).iloc[1] == pytest.approx(0.10)
    assert sma(pd.Series([10.0] * 10), 5).iloc[4] == pytest.approx(10.0)


def test_rsi_bounded():
    r = rsi_wilder(pd.Series(np.linspace(100, 150, 30)), 14).dropna()
    assert len(r) > 0 and r.max() <= 100 and r.min() >= 0


def test_build_features():
    n = 40
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.arange(n) * 0.5
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1e6},
        index=dates,
    )
    out = build_feature_frame(df)
    assert "rsi_14" in out.columns and "macd_hist" in out.columns

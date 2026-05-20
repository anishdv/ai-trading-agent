"""Manual pandas indicators — point-in-time only."""

from __future__ import annotations

import pandas as pd

from src.data import config

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean()


def returns(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(periods=window, fill_method=None)


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    out = pd.Series(index=close.index, dtype=float)
    mask_zero = (avg_loss == 0) | avg_loss.isna()
    out[mask_zero & (avg_gain > 0)] = 100.0
    out[mask_zero & (avg_gain == 0)] = 50.0
    valid = ~mask_zero
    rs = avg_gain[valid] / avg_loss[valid]
    out[valid] = 100.0 - (100.0 / (1.0 + rs))
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def rolling_return_std(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(fill_method=None).rolling(window, min_periods=window).std()


def volume_features(volume: pd.Series, window: int):
    vol_ma = volume.rolling(window, min_periods=window).mean()
    return vol_ma, volume / vol_ma.replace(0, pd.NA)


def build_feature_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv[OHLCV_COLUMNS].copy()
    close, volume = df["close"], df["volume"]
    for w in config.SMA_WINDOWS:
        df[f"sma_{w}"] = sma(close, w)
    for w in config.RET_WINDOWS:
        df[f"ret_{w}d"] = returns(close, w)
    df[f"rsi_{config.RSI_PERIOD}"] = rsi_wilder(close, config.RSI_PERIOD)
    m, s, h = macd(close, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)
    df["macd"], df["macd_signal"], df["macd_hist"] = m, s, h
    df[f"roll_std_{config.ROLLING_STD_WINDOW}"] = rolling_return_std(
        close, config.ROLLING_STD_WINDOW
    )
    vma, vr = volume_features(volume, config.VOL_MA_WINDOW)
    df[f"vol_ma_{config.VOL_MA_WINDOW}"] = vma
    df["vol_ratio"] = vr
    return df

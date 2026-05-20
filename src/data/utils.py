"""Data QA helpers."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.data import config

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
FEATURE_COLUMNS = [
    "sma_5", "sma_10", "ret_1d", "ret_3d", "rsi_14",
    "macd", "macd_signal", "macd_hist",
    f"roll_std_{config.ROLLING_STD_WINDOW}",
    f"vol_ma_{config.VOL_MA_WINDOW}", "vol_ratio",
]


def assert_ohlcv_schema(df: pd.DataFrame) -> None:
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Index must be unique and monotonic increasing")


def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not cols:
        return df
    before = len(df)
    out = df.dropna(subset=cols).copy()
    if before - len(out):
        logger.info("Dropped %d warmup rows", before - len(out))
    return out


def quality_report(df: pd.DataFrame, symbol: str = "") -> dict[str, Any]:
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    nan_total = int(df[feature_cols].isna().sum().sum()) if feature_cols else 0
    macd_ok = True
    if all(c in df.columns for c in ("macd", "macd_signal", "macd_hist")) and len(df):
        macd_ok = bool(
            (df["macd"] - df["macd_signal"] - df["macd_hist"]).abs().max() < 1e-9
        )
    rsi_min = rsi_max = None
    if "rsi_14" in df.columns and len(df):
        rsi_min, rsi_max = float(df["rsi_14"].min()), float(df["rsi_14"].max())
    return {
        "symbol": symbol,
        "row_count": len(df),
        "feature_nan_total": nan_total,
        "rsi_min": rsi_min,
        "rsi_max": rsi_max,
        "macd_hist_identity": macd_ok,
        "passed": len(df) > 0 and nan_total == 0 and macd_ok
            and (rsi_min is None or rsi_min >= -1e-6)
            and (rsi_max is None or rsi_max <= 100 + 1e-6),
    }


def ensure_dirs() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

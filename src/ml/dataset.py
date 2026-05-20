"""Time-based datasets from processed Parquet — no shuffle, no leakage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import WATCHLIST
from src.data.pipeline import processed_path
from src.ml.config import FEATURE_COLS, TRAIN_RATIO, VAL_RATIO


def _load_symbol(symbol: str) -> pd.DataFrame:
    path = processed_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data: {path}")
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df["symbol"] = symbol.upper()
    return df.sort_index()


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label at bar t: 1 if next-day return > 0, else 0.
    Uses forward return; drop last row per symbol (no label).
    """
    out = df.copy()
    out["fwd_ret_1d"] = out["close"].pct_change().shift(-1)
    out["label_up"] = (out["fwd_ret_1d"] > 0).astype(int)
    return out.dropna(subset=["fwd_ret_1d"])


def build_panel(symbols: list[str] | None = None) -> pd.DataFrame:
    symbols = symbols or WATCHLIST
    frames = [_load_symbol(s) for s in symbols]
    panel = pd.concat(frames, axis=0)
    panel = add_label(panel)
    missing = [c for c in FEATURE_COLS if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return panel.dropna(subset=FEATURE_COLS)


def time_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split by global date (no shuffle)."""
    df = df.sort_index()
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def xy(split: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = split[FEATURE_COLS].values
    y = split["label_up"].values
    return x, y

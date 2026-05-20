"""yfinance ingestion + Parquet cache."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.common.settings import settings
from src.data import config
from src.data.utils import OHLCV_COLUMNS, assert_ohlcv_schema, ensure_dirs

logger = logging.getLogger(__name__)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(
        columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        }
    )
    df = df[[c for c in OHLCV_COLUMNS if c in df.columns]].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=OHLCV_COLUMNS)


def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    kwargs = {
        "start": settings.start_date,
        "interval": "1d",
        "auto_adjust": True,
        "actions": False,
    }
    if settings.end_date:
        kwargs["end"] = settings.end_date
    raw = yf.Ticker(symbol).history(**kwargs)
    if raw.empty:
        raw = yf.download(
            symbol, start=settings.start_date, end=settings.end_date,
            interval="1d", auto_adjust=True, progress=False, threads=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.droplevel(1, axis=1)
    if raw.empty:
        raise ValueError(f"No data for {symbol}")
    df = _normalize(raw)
    assert_ohlcv_schema(df)
    return df


def raw_path(symbol: str) -> Path:
    return config.RAW_DIR / f"{symbol.upper()}.parquet"


def load_from_fixture(symbol: str) -> pd.DataFrame:
    path = config.FIXTURE_DIR / f"{symbol.upper()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run: python scripts/seed_fixtures.py ({path})")
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    assert_ohlcv_schema(df)
    return df


def load_or_fetch(symbol: str, force_refresh: bool = False) -> pd.DataFrame:
    if config.USE_FIXTURES:
        df = load_from_fixture(symbol)
        save_raw(df, symbol)
        return df
    path = raw_path(symbol)
    if path.exists() and not force_refresh:
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        assert_ohlcv_schema(df)
        return df
    df = fetch_ohlcv(symbol)
    save_raw(df, symbol)
    return df


def save_raw(df: pd.DataFrame, symbol: str) -> Path:
    ensure_dirs()
    path = raw_path(symbol)
    out = df.copy()
    out.index.name = "date"
    out.to_parquet(path)
    logger.info("Cached raw %s (%d rows)", symbol, len(out))
    return path

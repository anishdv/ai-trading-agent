"""Feature pipeline CLI."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.common.logging import setup_logging
from src.data import config
from src.data.config import WATCHLIST
from src.data.features import build_feature_frame
from src.data.loader import load_or_fetch
from src.data.utils import drop_warmup_rows, ensure_dirs, quality_report

logger = logging.getLogger(__name__)


def processed_path(symbol: str):
    return config.PROCESSED_DIR / f"{symbol.upper()}.parquet"


def run_symbol(symbol: str, force_refresh: bool = False):
    ohlcv = load_or_fetch(symbol, force_refresh=force_refresh)
    featured = build_feature_frame(ohlcv)
    processed = drop_warmup_rows(featured)
    report = quality_report(processed, symbol)
    if not report["passed"]:
        raise RuntimeError(f"QA failed {symbol}: {report}")
    ensure_dirs()
    path = processed_path(symbol)
    out = processed.copy()
    out.index.name = "date"
    out.to_parquet(path)
    logger.info("%s: %d rows -> %s", symbol, len(out), path)
    return path


def run_watchlist(symbols: list[str] | None = None, force_refresh: bool = False):
    symbols = symbols or WATCHLIST
    results = {}
    for sym in symbols:
        try:
            run_symbol(sym, force_refresh)
            df = pd.read_parquet(processed_path(sym))
            if "date" in df.columns:
                df = df.set_index("date")
            results[sym] = df
        except Exception:
            logger.exception("Failed %s", sym)
            if config.FAIL_FAST:
                raise
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-fixtures", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args(argv)
    if args.use_fixtures:
        config.USE_FIXTURES = True
    setup_logging()
    ensure_dirs()
    try:
        run_watchlist(force_refresh=args.force_refresh)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

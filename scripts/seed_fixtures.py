"""Generate offline OHLCV fixtures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.config import WATCHLIST  # noqa: E402
from src.data.utils import OHLCV_COLUMNS  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "raw"
N = 500


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in WATCHLIST:
        rng = np.random.default_rng(abs(hash(sym)) % (2**32))
        dates = pd.date_range("2022-01-03", periods=N, freq="B")
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, N)))
        high = close * (1 + rng.uniform(0.001, 0.02, N))
        low = close * (1 - rng.uniform(0.001, 0.02, N))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close,
             "volume": rng.integers(5_000_000, 50_000_000, N).astype(float)},
            index=dates,
        )
        df.index.name = "date"
        path = OUT / f"{sym}.parquet"
        df[OHLCV_COLUMNS].to_parquet(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

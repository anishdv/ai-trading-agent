"""Data-layer configuration."""

from pathlib import Path

from src.common.settings import PROJECT_ROOT, settings

WATCHLIST: list[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "JPM",
    "XOM",
    "JNJ",
    "NVDA",
    "META",
    "SPY",
]

SMA_WINDOWS = [5, 10]
RET_WINDOWS = [1, 3]
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ROLLING_STD_WINDOW = 20
VOL_MA_WINDOW = 20

USE_FIXTURES = False
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "raw"
RAW_DIR = settings.raw_dir
PROCESSED_DIR = settings.processed_dir
FAIL_FAST = True

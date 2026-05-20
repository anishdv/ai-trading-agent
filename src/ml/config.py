"""ML training configuration."""

from pathlib import Path

from src.common.settings import PROJECT_ROOT

# Feature columns for model (no OHLCV raw prices — scale-invariant)
FEATURE_COLS = [
    "ret_1d",
    "ret_3d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "roll_std_20",
    "vol_ratio",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# remainder = test

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "direction_model.joblib"
METADATA_FILE = MODELS_DIR / "direction_model_meta.json"

"""Load model and produce probabilistic predictions."""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

import joblib
import pandas as pd

from src.common.schemas import ModelPrediction
from src.data.pipeline import processed_path
from src.ml.config import FEATURE_COLS, METADATA_FILE, MODEL_FILE


def load_metadata() -> dict:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"No model metadata at {METADATA_FILE}. Run: python scripts/train_model.py"
        )
    return json.loads(METADATA_FILE.read_text())


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"No model at {MODEL_FILE}. Run: python scripts/train_model.py")
    return joblib.load(MODEL_FILE)


def get_model_prediction(symbol: str) -> ModelPrediction:
    """P(up) for the latest bar — features at t predict label t→t+1."""
    path = processed_path(symbol)
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    row = df.iloc[-1]
    x = row[FEATURE_COLS].values.reshape(1, -1)

    pipe = load_model()
    meta = load_metadata()
    prob_up = float(pipe.predict_proba(x)[0, 1])

    return ModelPrediction(
        symbol=symbol.upper(),
        as_of=df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc),
        prob_up=prob_up,
        prob_down=1.0 - prob_up,
        model_version=meta["model_version"],
        metadata={"holdout_brier": meta.get("test_metrics", {}).get("brier")},
    )

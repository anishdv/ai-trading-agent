"""Train baseline direction classifier."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.ml.config import METADATA_FILE, MODEL_FILE, MODELS_DIR
from src.ml.dataset import build_panel, time_split, xy
from src.ml.metrics import evaluate_probs

logger = logging.getLogger(__name__)


def train_model() -> dict:
    panel = build_panel()
    train_df, val_df, test_df = time_split(panel)
    x_train, y_train = xy(train_df)
    x_val, y_val = xy(val_df)
    x_test, y_test = xy(test_df)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(x_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_FILE)

    val_probs = pipe.predict_proba(x_val)[:, 1]
    test_probs = pipe.predict_proba(x_test)[:, 1]

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_version": hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()[:12],
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "val_metrics": evaluate_probs(y_val, val_probs),
        "test_metrics": evaluate_probs(y_test, test_probs),
        "label": "P(next_day_return > 0)",
    }
    METADATA_FILE.write_text(json.dumps(meta, indent=2))
    logger.info("Model saved %s", MODEL_FILE)
    logger.info("Val metrics: %s", meta["val_metrics"])
    logger.info("Test metrics: %s", meta["test_metrics"])
    return meta

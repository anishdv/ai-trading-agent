"""Calibration and scoring metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score


def evaluate_probs(y_true: np.ndarray, prob_up: np.ndarray) -> dict[str, float | None]:
    pred = (prob_up >= 0.5).astype(int)
    out: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob_up)),
        "n_samples": float(len(y_true)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob_up))
    except ValueError:
        out["roc_auc"] = None
    return out

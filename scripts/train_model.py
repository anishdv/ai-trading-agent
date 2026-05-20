#!/usr/bin/env python3
"""Train direction model on processed watchlist data."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.logging import setup_logging
from src.ml.train import train_model


def main():
    setup_logging()
    meta = train_model()
    print("\n=== Training complete ===")
    print(f"Model version: {meta['model_version']}")
    print(f"Val:  {meta['val_metrics']}")
    print(f"Test: {meta['test_metrics']}")
    print("\n>>> MANUAL: Review test Brier (lower is better). Target < 0.25 is decent for daily bars.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

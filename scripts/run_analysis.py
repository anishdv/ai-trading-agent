#!/usr/bin/env python3
"""Phase 1 CLI: signals + risk-checked draft proposal for one symbol."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.signals import compute_signals
from src.common.logging import setup_logging
from src.common.schemas import Side, SignalDirection
from src.data.pipeline import processed_path
from src.ml.predict import get_model_prediction
from src.risk.proposals import propose_trade
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run deterministic analysis + draft proposal")
    parser.add_argument("symbol", help="Ticker e.g. AAPL")
    parser.add_argument("--qty", type=float, default=None)
    parser.add_argument("--no-ml", action="store_true", help="Skip ML prediction")
    args = parser.parse_args()
    setup_logging()

    path = processed_path(args.symbol)
    if not path.exists():
        print(
            "\n>>> MANUAL STEP: Run the data pipeline first:\n"
            "    python -m src.data.pipeline --use-fixtures\n"
            "    (or without --use-fixtures if yfinance works)\n"
        )
        return 1

    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    report = compute_signals(args.symbol, df)
    print("\n=== Signal Report ===")
    print(report.model_dump_json(indent=2))

    prediction = None
    if not args.no_ml:
        try:
            prediction = get_model_prediction(args.symbol)
            print("\n=== ML Prediction ===")
            print(prediction.model_dump_json(indent=2))
        except FileNotFoundError as e:
            print(f"\n>>> MANUAL STEP: Train model first:\n    python scripts/train_model.py\n    ({e})")

    # Combine rules + ML: require agreement for strong signal, else fall back to rules
    side = None
    if report.direction == SignalDirection.BULLISH:
        side = Side.BUY
    elif report.direction == SignalDirection.BEARISH:
        side = Side.SELL

    if prediction and side:
        ml_bull = prediction.prob_up >= 0.55
        ml_bear = prediction.prob_up <= 0.45
        if side == Side.BUY and not ml_bull:
            print("\n>>> Rules bullish but ML not confident (P(up)<0.55) — no proposal.")
            return 0
        if side == Side.SELL and not ml_bear:
            print("\n>>> Rules bearish but ML not confident (P(up)>0.45) — no proposal.")
            return 0

    if side is None:
        print("\n>>> Neutral signals — no proposal created.")
        return 0

    last_price = float(df.iloc[-1]["close"])
    rationale = f"Rules {report.direction.value} ({report.composite_score:.2f})"
    if prediction:
        rationale += f"; ML P(up)={prediction.prob_up:.2f}"
    proposal = propose_trade(
        symbol=args.symbol,
        side=side,
        quantity=args.qty,
        rationale=rationale,
        signal_report=report,
        model_prediction=prediction,
        last_price=last_price,
    )

    print("\n=== Trade Proposal (draft only) ===")
    print(proposal.model_dump_json(indent=2))
    print(f"\n>>> Saved to data/proposals/{proposal.proposal_id}.json")
    print(">>> No broker or WhatsApp in Phase 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

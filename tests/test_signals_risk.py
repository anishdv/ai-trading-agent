from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.analysis.signals import compute_signals
from src.common.schemas import RiskStatus, Side, SignalDirection
from src.risk.engine import evaluate_proposal
from src.risk.portfolio import mock_portfolio
from src.risk.proposals import propose_trade
from src.common.schemas import TradeProposal


def _features(n=60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    t = np.arange(n)
    close = 100 + t * 0.8
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1e6},
        index=dates,
    )
    from src.data.features import build_feature_frame
    from src.data.utils import drop_warmup_rows
    return drop_warmup_rows(build_feature_frame(df))


def test_signals_bullish_trend():
    df = _features()
    # force bullish last row
    df.iloc[-1, df.columns.get_loc("close")] = df.iloc[-1]["sma_10"] + 5
    df.iloc[-1, df.columns.get_loc("sma_5")] = df.iloc[-1]["close"] - 1
    report = compute_signals("TEST", df)
    assert report.direction in (SignalDirection.BULLISH, SignalDirection.NEUTRAL)


def test_risk_blocks_oversized_buy():
    p = TradeProposal(
        proposal_id="t1", symbol="AAPL", side=Side.BUY, quantity=10_000,
        created_at=datetime.now(timezone.utc),
    )
    v = evaluate_proposal(p, mock_portfolio(equity=100_000, cash=50_000), last_price=200.0)
    assert v.status == RiskStatus.BLOCK


def test_propose_trade_persists(tmp_path, monkeypatch):
    def _fake_dir():
        tmp_path.mkdir(parents=True, exist_ok=True)
        return tmp_path
    monkeypatch.setattr("src.risk.proposals._dir", _fake_dir)
    df = _features()
    report = compute_signals("AAPL", df)
    prop = propose_trade("AAPL", Side.BUY, quantity=5, signal_report=report, last_price=float(df.iloc[-1]["close"]))
    assert (tmp_path / f"{prop.proposal_id}.json").exists()

"""Deterministic rule-based signals from processed features."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.common.schemas import SignalDirection, SignalFlag, SignalReport


def _latest_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("Empty feature frame")
    return df.iloc[-1]


def compute_signals(symbol: str, df: pd.DataFrame) -> SignalReport:
    """Evaluate rules on the last bar only (point-in-time)."""
    row = _latest_row(df)
    flags: list[SignalFlag] = []
    scores: list[float] = []

    close = float(row["close"])
    sma5 = float(row["sma_5"])
    sma10 = float(row["sma_10"])
    if close > sma5 > sma10:
        flags.append(SignalFlag(name="trend_aligned_up", direction=SignalDirection.BULLISH, strength=0.7, detail="close > sma5 > sma10"))
        scores.append(0.5)
    elif close < sma5 < sma10:
        flags.append(SignalFlag(name="trend_aligned_down", direction=SignalDirection.BEARISH, strength=0.7, detail="close < sma5 < sma10"))
        scores.append(-0.5)

    rsi = float(row["rsi_14"])
    if rsi < 30:
        flags.append(SignalFlag(name="rsi_oversold", direction=SignalDirection.BULLISH, strength=0.6, detail=f"rsi={rsi:.1f}"))
        scores.append(0.35)
    elif rsi > 70:
        flags.append(SignalFlag(name="rsi_overbought", direction=SignalDirection.BEARISH, strength=0.6, detail=f"rsi={rsi:.1f}"))
        scores.append(-0.35)

    macd_hist = float(row["macd_hist"])
    if macd_hist > 0:
        flags.append(SignalFlag(name="macd_hist_positive", direction=SignalDirection.BULLISH, strength=0.4, detail=f"hist={macd_hist:.4f}"))
        scores.append(0.2)
    elif macd_hist < 0:
        flags.append(SignalFlag(name="macd_hist_negative", direction=SignalDirection.BEARISH, strength=0.4, detail=f"hist={macd_hist:.4f}"))
        scores.append(-0.2)

    ret3 = float(row["ret_3d"])
    if ret3 > 0.02:
        flags.append(SignalFlag(name="momentum_3d_up", direction=SignalDirection.BULLISH, strength=0.5, detail=f"ret_3d={ret3:.2%}"))
        scores.append(0.25)
    elif ret3 < -0.02:
        flags.append(SignalFlag(name="momentum_3d_down", direction=SignalDirection.BEARISH, strength=0.5, detail=f"ret_3d={ret3:.2%}"))
        scores.append(-0.25)

    composite = sum(scores) / len(scores) if scores else 0.0
    composite = max(-1.0, min(1.0, composite))
    if composite > 0.15:
        direction = SignalDirection.BULLISH
    elif composite < -0.15:
        direction = SignalDirection.BEARISH
    else:
        direction = SignalDirection.NEUTRAL

    return SignalReport(
        symbol=symbol.upper(),
        as_of=df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc),
        composite_score=composite,
        direction=direction,
        flags=flags,
        metadata={"close": close, "rsi_14": rsi},
    )

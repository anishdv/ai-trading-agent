"""MCP-style tool registry — validated handlers, audit trail."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from src.analysis.signals import compute_signals
from src.common.schemas import Side
from src.data.config import WATCHLIST
from src.data.pipeline import processed_path
from src.ml.predict import get_model_prediction
from src.risk import config as risk_config
from src.risk.portfolio import mock_portfolio
from src.risk.proposals import propose_trade

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    read_only: bool
    handler: Callable[..., dict[str, Any]]


def _load_features(symbol: str, lookback: int = 30) -> dict[str, Any]:
    path = processed_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"No processed data for {symbol}")
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    df = df.tail(lookback)
    last = df.iloc[-1]
    numeric_cols = df.select_dtypes(include="number").columns
    return {
        "symbol": symbol.upper(),
        "bars": len(df),
        "as_of": str(df.index[-1]),
        "last_bar": {k: float(last[k]) for k in numeric_cols},
    }


def _get_watchlist() -> dict[str, Any]:
    return {"watchlist": list(WATCHLIST)}


def _get_signals(symbol: str) -> dict[str, Any]:
    path = processed_path(symbol)
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    report = compute_signals(symbol, df)
    return json.loads(report.model_dump_json())


def _get_model_prediction(symbol: str) -> dict[str, Any]:
    pred = get_model_prediction(symbol)
    return json.loads(pred.model_dump_json())


def _get_portfolio() -> dict[str, Any]:
    return json.loads(mock_portfolio().model_dump_json())


def _get_risk_limits() -> dict[str, Any]:
    return {
        "max_position_pct": risk_config.MAX_POSITION_PCT,
        "max_positions": risk_config.MAX_PORTFOLIO_POSITIONS,
        "min_cash_reserve_pct": risk_config.MIN_CASH_RESERVE_PCT,
        "default_order_qty": risk_config.DEFAULT_ORDER_QTY,
    }


def _propose_trade(
    symbol: str,
    side: str,
    quantity: float,
    rationale: str,
    last_price: float | None = None,
) -> dict[str, Any]:
    path = processed_path(symbol)
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    price = last_price or float(df.iloc[-1]["close"])
    proposal = propose_trade(
        symbol=symbol,
        side=Side(side.upper()),
        quantity=quantity,
        rationale=rationale,
        last_price=price,
    )
    return json.loads(proposal.model_dump_json())


TOOLS: list[Tool] = [
    Tool("get_watchlist", "List configured watchlist symbols", True, lambda: _get_watchlist()),
    Tool("get_ohlcv_features", "Latest OHLCV + features snapshot", True, lambda symbol, lookback_days=30: _load_features(symbol, lookback_days)),
    Tool("get_signals", "Deterministic signal report", True, lambda symbol: _get_signals(symbol)),
    Tool("get_model_prediction", "ML P(up) next day", True, lambda symbol: _get_model_prediction(symbol)),
    Tool("get_portfolio", "Current portfolio snapshot (mock)", True, lambda: _get_portfolio()),
    Tool("get_risk_limits", "Risk engine limits", True, lambda: _get_risk_limits()),
    Tool(
        "propose_trade",
        "Create risk-checked draft proposal (no execution)",
        False,
        lambda symbol, side, quantity, rationale, last_price=None: _propose_trade(
            symbol, side, quantity, rationale, last_price
        ),
    ),
]

_TOOL_MAP = {t.name: t for t in TOOLS}


def list_tools() -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "read_only": t.read_only}
        for t in TOOLS
    ]


def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    if name not in _TOOL_MAP:
        raise ValueError(f"Unknown tool: {name}")
    tool = _TOOL_MAP[name]
    args = arguments or {}
    logger.info("tool=%s args=%s", name, args)
    if name == "get_watchlist":
        return tool.handler()
    if name == "get_ohlcv_features":
        return tool.handler(args["symbol"], args.get("lookback_days", 30))
    if name in ("get_signals", "get_model_prediction"):
        return tool.handler(args["symbol"])
    if name in ("get_portfolio", "get_risk_limits"):
        return tool.handler()
    if name == "propose_trade":
        return tool.handler(
            args["symbol"],
            args["side"],
            args["quantity"],
            args["rationale"],
            args.get("last_price"),
        )
    raise ValueError(f"Unhandled tool: {name}")

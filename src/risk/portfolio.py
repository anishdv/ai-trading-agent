"""Portfolio snapshot — mock for Phase 1; Alpaca in Phase 4."""

from __future__ import annotations

from datetime import datetime, timezone

from src.common.schemas import PortfolioPosition, PortfolioSnapshot


def mock_portfolio(equity: float = 100_000.0, cash: float = 50_000.0) -> PortfolioSnapshot:
    """Paper-style starting portfolio for development."""
    return PortfolioSnapshot(
        as_of=datetime.now(timezone.utc),
        cash=cash,
        equity=equity,
        positions=[
            PortfolioPosition(symbol="SPY", quantity=50, market_value=25_000.0),
        ],
    )

"""Pre-trade risk checks — deterministic, no LLM."""

from __future__ import annotations

from src.common.schemas import PortfolioSnapshot, RiskStatus, RiskVerdict, Side, TradeProposal
from src.risk import config as risk_config


def evaluate_proposal(
    proposal: TradeProposal,
    portfolio: PortfolioSnapshot,
    last_price: float,
) -> RiskVerdict:
    reasons: list[str] = []
    status = RiskStatus.ALLOW
    max_qty: float | None = proposal.quantity

    if proposal.quantity <= 0:
        return RiskVerdict(status=RiskStatus.BLOCK, reasons=["quantity must be positive"])

    notional = proposal.quantity * last_price
    max_notional = portfolio.equity * risk_config.MAX_POSITION_PCT

    existing = next((p for p in portfolio.positions if p.symbol == proposal.symbol), None)
    existing_notional = existing.market_value if existing else 0.0
    total_exposure = existing_notional + (notional if proposal.side == Side.BUY else 0)

    if total_exposure > max_notional:
        status = RiskStatus.BLOCK
        reasons.append(
            f"position would be ${total_exposure:,.0f} > max ${max_notional:,.0f} "
            f"({risk_config.MAX_POSITION_PCT:.0%} of equity)"
        )

    if proposal.side == Side.BUY:
        cash_after = portfolio.cash - notional
        min_cash = portfolio.equity * risk_config.MIN_CASH_RESERVE_PCT
        if cash_after < min_cash:
            status = RiskStatus.BLOCK
            reasons.append(f"insufficient cash after buy (need reserve {risk_config.MIN_CASH_RESERVE_PCT:.0%})")
        allowed_qty = max(0.0, (portfolio.cash - min_cash) / last_price)
        if status != RiskStatus.BLOCK and proposal.quantity > allowed_qty:
            status = RiskStatus.MODIFY
            max_qty = float(int(allowed_qty))
            reasons.append(f"qty reduced to {max_qty:.0f} to respect cash reserve")

    if proposal.side == Side.SELL:
        held = existing.quantity if existing else 0.0
        if proposal.quantity > held:
            status = RiskStatus.BLOCK
            reasons.append(f"cannot sell {proposal.quantity} > held {held}")

    if proposal.side == Side.BUY and not existing:
        if len(portfolio.positions) >= risk_config.MAX_PORTFOLIO_POSITIONS:
            status = RiskStatus.BLOCK
            reasons.append(f"max positions ({risk_config.MAX_PORTFOLIO_POSITIONS}) reached")

    return RiskVerdict(status=status, reasons=reasons, max_allowed_qty=max_qty)

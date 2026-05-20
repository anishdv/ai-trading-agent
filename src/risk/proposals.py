"""Draft trade proposal store — no execution."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.common.schemas import (
    ModelPrediction,
    ProposalStatus,
    RiskStatus,
    Side,
    SignalReport,
    TradeProposal,
)
from src.common.settings import settings
from src.risk import config as risk_config
from src.risk.engine import evaluate_proposal
from src.risk.portfolio import mock_portfolio


def _dir() -> Path:
    d = settings.proposals_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def propose_trade(
    symbol: str,
    side: Side,
    quantity: float | None = None,
    rationale: str = "",
    signal_report: SignalReport | None = None,
    model_prediction=None,
    last_price: float = 100.0,
    portfolio=None,
) -> TradeProposal:
    """
    Create a draft proposal after risk check.
    Does NOT submit to broker or WhatsApp (Phase 4).
    """
    portfolio = portfolio or mock_portfolio()
    qty = quantity if quantity is not None else risk_config.DEFAULT_ORDER_QTY

    proposal = TradeProposal(
        proposal_id=str(uuid.uuid4())[:8],
        symbol=symbol.upper(),
        side=side,
        quantity=qty,
        rationale=rationale,
        signal_report=signal_report,
        model_prediction=model_prediction,
        status=ProposalStatus.DRAFT,
        created_at=datetime.now(timezone.utc),
    )

    verdict = evaluate_proposal(proposal, portfolio, last_price)
    proposal.risk_verdict = verdict

    if verdict.status == RiskStatus.BLOCK:
        proposal.status = ProposalStatus.REJECTED
    elif verdict.status == RiskStatus.MODIFY and verdict.max_allowed_qty is not None:
        proposal.quantity = verdict.max_allowed_qty
        proposal.status = ProposalStatus.DRAFT
    else:
        proposal.status = ProposalStatus.DRAFT

    path = _dir() / f"{proposal.proposal_id}.json"
    path.write_text(proposal.model_dump_json(indent=2))
    return proposal

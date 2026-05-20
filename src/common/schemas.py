"""Pydantic contracts shared across layers."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class SignalDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class RiskStatus(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    MODIFY = "MODIFY"


class ProposalStatus(str, Enum):
    DRAFT = "DRAFT"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"


class SignalFlag(BaseModel):
    name: str
    direction: SignalDirection
    strength: float = Field(ge=0.0, le=1.0)
    detail: str = ""


class SignalReport(BaseModel):
    symbol: str
    as_of: datetime
    composite_score: float = Field(ge=-1.0, le=1.0)
    direction: SignalDirection
    flags: list[SignalFlag]
    metadata: dict[str, Any] = Field(default_factory=dict)


class PortfolioPosition(BaseModel):
    symbol: str
    quantity: float
    market_value: float = 0.0


class PortfolioSnapshot(BaseModel):
    as_of: datetime
    cash: float
    equity: float
    positions: list[PortfolioPosition] = Field(default_factory=list)


class RiskVerdict(BaseModel):
    status: RiskStatus
    reasons: list[str] = Field(default_factory=list)
    max_allowed_qty: float | None = None


class ModelPrediction(BaseModel):
    symbol: str
    as_of: datetime
    prob_up: float = Field(ge=0.0, le=1.0)
    prob_down: float = Field(ge=0.0, le=1.0)
    model_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TradeProposal(BaseModel):
    proposal_id: str
    symbol: str
    side: Side
    quantity: float
    order_type: str = "market"
    status: ProposalStatus = ProposalStatus.DRAFT
    rationale: str = ""
    signal_report: SignalReport | None = None
    model_prediction: ModelPrediction | None = None
    risk_verdict: RiskVerdict | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

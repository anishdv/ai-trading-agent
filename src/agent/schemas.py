"""Agent output contracts."""

from pydantic import BaseModel, Field

from src.common.schemas import ModelPrediction, SignalReport, TradeProposal


class AgentRecommendation(BaseModel):
    action: str = Field(description="BUY, SELL, or HOLD")
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str
    risks: list[str] = Field(default_factory=list)
    dissent: str = ""


class AgentCycleResult(BaseModel):
    symbol: str
    recommendation: AgentRecommendation
    signal_report: SignalReport | None = None
    model_prediction: ModelPrediction | None = None
    proposal: TradeProposal | None = None
    tool_trace: list[str] = Field(default_factory=list)
    llm_used: bool = False

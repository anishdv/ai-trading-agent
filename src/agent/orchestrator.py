"""Agent cycle: tools -> context -> LLM -> optional draft proposal."""

from __future__ import annotations

import json
import logging

from src.agent.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.agent.schemas import AgentCycleResult, AgentRecommendation
from src.agent.tools import execute_tool
from src.common.schemas import Side
from src.common.settings import settings
from src.risk import config as risk_config

logger = logging.getLogger(__name__)


def gather_context(symbol: str) -> tuple[dict, list[str]]:
    trace: list[str] = []
    ctx: dict = {"symbol": symbol.upper()}

    for name, key, args in [
        ("get_signals", "signals", {"symbol": symbol}),
        ("get_model_prediction", "ml_prediction", {"symbol": symbol}),
        ("get_ohlcv_features", "market", {"symbol": symbol, "lookback_days": 30}),
        ("get_portfolio", "portfolio", {}),
        ("get_risk_limits", "risk_limits", {}),
    ]:
        try:
            ctx[key] = execute_tool(name, args)
            trace.append(name)
        except FileNotFoundError as e:
            ctx[key] = {"error": str(e)}
            trace.append(f"{name}:skipped")

    return ctx, trace


def _rule_based_recommendation(ctx: dict) -> AgentRecommendation:
    """Fallback when no OpenAI key — transparent rules."""
    signals = ctx.get("signals", {})
    ml = ctx.get("ml_prediction", {})
    direction = signals.get("direction", "neutral")
    prob_up = ml.get("prob_up", 0.5)

    if direction == "bullish" and prob_up >= 0.55:
        return AgentRecommendation(
            action="BUY",
            confidence=min(0.85, prob_up),
            thesis="Bullish rules aligned with ML P(up) >= 0.55",
            risks=["Daily bar latency", "Model near random on weak features"],
            dissent="Bearish macro not modeled here",
        )
    if direction == "bearish" and prob_up <= 0.45:
        return AgentRecommendation(
            action="SELL",
            confidence=min(0.85, 1 - prob_up),
            thesis="Bearish rules aligned with ML P(up) <= 0.45",
            risks=["Short exposure may be blocked by mock portfolio"],
            dissent="Oversold bounce possible per RSI",
        )
    return AgentRecommendation(
        action="HOLD",
        confidence=0.5,
        thesis="Insufficient agreement between rules and ML",
        risks=["Signal noise on daily timeframe"],
        dissent="",
    )


def _llm_recommendation(symbol: str, ctx: dict) -> AgentRecommendation:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    user_msg = USER_TEMPLATE.format(
        symbol=symbol,
        context_json=json.dumps(ctx, indent=2, default=str),
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    return AgentRecommendation.model_validate(data)


def run_agent_cycle(symbol: str, quantity: float | None = None) -> AgentCycleResult:
    ctx, trace = gather_context(symbol)
    llm_used = bool(settings.openai_api_key)

    if llm_used:
        try:
            rec = _llm_recommendation(symbol, ctx)
        except Exception:
            logger.exception("LLM failed; falling back to rules")
            rec = _rule_based_recommendation(ctx)
            llm_used = False
    else:
        rec = _rule_based_recommendation(ctx)

    proposal = None
    if rec.action in ("BUY", "SELL"):
        qty = quantity or risk_config.DEFAULT_ORDER_QTY
        last_price = ctx.get("market", {}).get("last_bar", {}).get("close")
        proposal_data = execute_tool(
            "propose_trade",
            {
                "symbol": symbol,
                "side": rec.action,
                "quantity": qty,
                "rationale": f"Agent: {rec.thesis}",
                "last_price": last_price,
            },
        )
        from src.common.schemas import TradeProposal
        proposal = TradeProposal.model_validate(proposal_data)

    signals = ctx.get("signals")
    ml = ctx.get("ml_prediction")
    from src.common.schemas import ModelPrediction, SignalReport

    return AgentCycleResult(
        symbol=symbol.upper(),
        recommendation=rec,
        signal_report=SignalReport.model_validate(signals) if signals and "error" not in signals else None,
        model_prediction=ModelPrediction.model_validate(ml) if ml and "error" not in ml else None,
        proposal=proposal,
        tool_trace=trace,
        llm_used=llm_used,
    )

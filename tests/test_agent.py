from unittest.mock import patch

import pytest

from src.agent.orchestrator import gather_context, run_agent_cycle
from src.agent.tools import execute_tool, list_tools


def test_tool_registry():
    names = {t["name"] for t in list_tools()}
    assert "get_signals" in names
    assert "propose_trade" in names


def test_gather_context_requires_data():
    with pytest.raises(FileNotFoundError):
        execute_tool("get_signals", {"symbol": "ZZZZ"})


@patch("src.agent.orchestrator.settings")
def test_agent_cycle_fallback_no_api_key(mock_settings, tmp_path, monkeypatch):
    mock_settings.openai_api_key = ""
    monkeypatch.setattr("src.risk.proposals._dir", lambda: tmp_path)

    # use fixture path if processed exists
    from src.data.pipeline import processed_path
    if not processed_path("AAPL").exists():
        pytest.skip("processed AAPL missing")

    result = run_agent_cycle("AAPL")
    assert result.symbol == "AAPL"
    assert result.recommendation.action in ("BUY", "SELL", "HOLD")
    assert result.llm_used is False

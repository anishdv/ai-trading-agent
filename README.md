# AI Trading Agent System

Intelligent trading **assistant** with mandatory human approval (Twilio WhatsApp) before paper execution.

## Phase 3 (current): LLM agent + tools

- MCP-style tool registry (`get_signals`, `get_model_prediction`, `propose_trade`, …)
- Orchestrator gathers context, then LLM (or rule fallback) recommends BUY/SELL/HOLD
- Draft proposals only — no WhatsApp / execution

```bash
python scripts/run_agent.py AAPL
```

## Phase 2: ML predictions

- Time-based train/val/test splits (no shuffle)
- Logistic regression: P(next-day return > 0)
- Predictions feed analysis CLI (rules + ML agreement)

## Phase 1: Deterministic analysis + risk

- Market data pipeline → features → Parquet
- Rule-based signal engine
- Portfolio-aware risk engine
- Trade proposals stored as drafts only (no execution)

## Quick start

```bash
cd ai-trading-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m src.data.pipeline --use-fixtures   # or without flag for yfinance
python scripts/train_model.py
python scripts/run_analysis.py AAPL
pytest
```

## Manual steps (highlighted in docs / CLI output)

See project wiki in chat — key gates:

1. **Setup** — venv + `.env`
2. **Data** — run pipeline (fixtures or yfinance)
3. **Analysis** — `run_analysis.py SYMBOL`
4. **Twilio / Alpaca** — not required until Phase 4

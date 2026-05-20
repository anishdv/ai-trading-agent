#!/usr/bin/env python3
"""Phase 3: LLM agent cycle with MCP-style tools."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.orchestrator import run_agent_cycle
from src.common.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run agent analysis cycle")
    parser.add_argument("symbol", help="Ticker e.g. AAPL")
    parser.add_argument("--qty", type=float, default=None)
    args = parser.parse_args()
    setup_logging()

    try:
        result = run_agent_cycle(args.symbol, quantity=args.qty)
    except FileNotFoundError:
        print(
            "\n>>> MANUAL STEP: Build data + train model:\n"
            "    python -m src.data.pipeline --use-fixtures\n"
            "    python scripts/train_model.py\n"
        )
        return 1

    print("\n=== Agent Cycle ===")
    print(result.model_dump_json(indent=2))
    if not result.llm_used:
        print(
            "\n>>> MANUAL (optional): Add OPENAI_API_KEY to .env for LLM reasoning.\n"
            "    Ran rule-based fallback (no API cost)."
        )
    if result.proposal:
        print(f"\n>>> Draft proposal: data/proposals/{result.proposal.proposal_id}.json")
    print(">>> No WhatsApp / broker execution (Phase 4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

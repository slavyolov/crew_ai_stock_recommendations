from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from colab_stock_crew.crew import StockAnalysisCrew


def validate_env() -> None:
    needed = ["OPENAI_API_KEY", "SERPER_API_KEY", "SEC_API_KEY"]
    missing = [k for k in needed if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True, help="Ticker, e.g. NVDA")
    p.add_argument("--company", required=True, help="Company name, e.g. NVIDIA")
    return p.parse_args()


def main():
    args = parse_args()
    validate_env()
    inputs = {"ticker": args.ticker.upper(), "company": args.company}
    result = StockAnalysisCrew().crew().kickoff(inputs=inputs)
    print(result)


if __name__ == "__main__":
    main()

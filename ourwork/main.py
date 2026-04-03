"""Minimal entry: feed question, get JSON result."""

import json
import os
import sys
from typing import Any, Dict

from agent import KimiCalculusAgent


def run(question: str, strategy: str | None = None) -> Dict[str, str]:
    agent = KimiCalculusAgent()
    return agent.solve(question, strategy=strategy)


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else None
    if not question:
        if sys.stdin.isatty():
            print("请输入微积分题目（Ctrl+D 结束）：")
        question = sys.stdin.read().strip()

    if not question:
        print("未输入题目，退出。")
        return

    strategy = os.getenv("AGENT_STRATEGY")
    result = run(question, strategy=strategy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

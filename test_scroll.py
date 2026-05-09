"""Test VLM scroll direction judgment + actual scroll execution.

Usage:
    uv run python test_scroll.py "查看更多内容"
    uv run python test_scroll.py "滚动查看更早的月份"
"""

import sys
from pathlib import Path

from policy_expr.perception import SCKSession
from policy_expr.policies import StructuredOutputPolicy
from policy_expr.schemas import Observation
from policy_expr.executor import ActionExecutor


def main():
    instruction = sys.argv[1] if len(sys.argv) > 1 else "滚动查看更多内容"

    sck = SCKSession()
    sck.start()
    png = sck.screenshot()
    print(f"Screenshot: {len(png) // 1024} KB")

    obs = Observation(png_bytes=png, source="live")
    policy = StructuredOutputPolicy()

    print(f"\n指令: {instruction}")
    print("VLM 决策中...\n")

    decision = policy.decide(obs, instruction)
    action = decision.action

    print(f"action_type: {action.action_type}")
    print(f"direction:   {action.direction}")
    print(f"x:           {action.x}")
    print(f"y:           {action.y}")
    print(f"description: {action.description}")

    out = Path("/tmp/scroll_test")
    out.mkdir(exist_ok=True)
    (out / "current.png").write_bytes(png)

    if action.action_type == "scroll":
        print("\n--- 触发滚动 ---")
        executor = ActionExecutor(phone=sck)
        executor.execute(decision)


if __name__ == "__main__":
    main()

"""CLI runner for single-turn policy experiments."""

import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from policy_expr.executor import ActionExecutor
from policy_expr.perception import LivePerception, LivePhoneSession, SCREENSHOT
from policy_expr.policies import StructuredOutputPolicy
from policy_expr.policies.base import Policy
from policy_expr.visualize import print_decision

POLICIES: dict[str, type[Policy]] = {
    StructuredOutputPolicy.name: StructuredOutputPolicy,
}


def build_policy(name: str) -> Policy:
    try:
        policy_cls = POLICIES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(POLICIES))
        raise ValueError(f"未知策略 {name!r}，可选：{choices}") from exc
    return policy_cls()


def run_once(prompt: str, policy: Policy) -> None:
    with LivePhoneSession() as phone:
        perception = LivePerception(phone)
        observation = perception.observe()

        print("分析中...")
        decision = policy.decide(observation, prompt)
        print_decision(decision, observation.png_bytes)

        executed = ActionExecutor(phone).execute(decision)
        if not executed:
            return

        time.sleep(1.5)
        print("点击后截图...")
        after_bytes = phone.screenshot()
        after_path = SCREENSHOT.parent / "screenshot_after.png"
        after_path.write_bytes(after_bytes)
        print(f"已保存: {after_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="测试智能单轮策略决策效果")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="打开微信",
        help="目标指令，如「打开微信」「发一条朋友圈」",
    )
    parser.add_argument(
        "--policy",
        default=StructuredOutputPolicy.name,
        choices=sorted(POLICIES),
        help="要测试的策略模块",
    )
    args = parser.parse_args()

    policy = build_policy(args.policy)
    run_once(args.prompt, policy)


if __name__ == "__main__":
    main()

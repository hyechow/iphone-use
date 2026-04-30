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


def run_continue(prompt: str, policy: Policy, context_path: Path) -> None:
    context = policy.load_context(context_path, prompt)
    policy.save_context(context_path, context)
    print(f"Context : {context_path}")
    print(f"Goal    : {context.goal}")
    print(f"Turns   : {len(context.turns)}")

    with LivePhoneSession() as phone:
        perception = LivePerception(phone)
        executor = ActionExecutor(phone)

        while True:
            turn_no = len(context.turns) + 1
            print(f"\n--- Turn {turn_no} ---")
            observation = perception.observe()

            print("分析中...")
            decision = policy.decide_with_context(observation, context)
            print_decision(decision, observation.png_bytes)

            executed = executor.execute(decision)
            policy.append_turn(context, observation, decision, executed)
            policy.save_context(context_path, context)
            print(f"Context 已保存: {context_path}")
            if not executed:
                return

            time.sleep(1.5)
            print("动作后截图...")
            after_bytes = phone.screenshot()
            after_path = SCREENSHOT.parent / f"screenshot_after_turn_{turn_no}.png"
            after_path.write_bytes(after_bytes)
            print(f"已保存: {after_path}")

            answer = input("继续下一轮？[Enter继续 / q退出] ").strip().lower()
            if answer in {"q", "quit", "exit"}:
                return


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
    parser.add_argument(
        "--continue",
        dest="continue_path",
        type=Path,
        help="启用逐轮 ReAct 测试模式，并从指定 context.json 加载/保存上下文",
    )
    args = parser.parse_args()

    policy = build_policy(args.policy)
    if args.continue_path:
        run_continue(args.prompt, policy, args.continue_path)
    else:
        run_once(args.prompt, policy)


if __name__ == "__main__":
    main()

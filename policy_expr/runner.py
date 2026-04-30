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
from policy_expr.schemas import Observation, PolicyDecision, TurnValidation
from policy_expr.validators import SimpleLLMValidator, Validator
from policy_expr.visualize import print_decision

POLICIES: dict[str, type[Policy]] = {
    StructuredOutputPolicy.name: StructuredOutputPolicy,
}
VALIDATORS: dict[str, type[Validator]] = {
    SimpleLLMValidator.name: SimpleLLMValidator,
}


def build_policy(name: str) -> Policy:
    try:
        policy_cls = POLICIES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(POLICIES))
        raise ValueError(f"未知策略 {name!r}，可选：{choices}") from exc
    return policy_cls()


def build_validator(name: str | None) -> Validator | None:
    if name in (None, "none"):
        return None
    try:
        validator_cls = VALIDATORS[name]
    except KeyError as exc:
        choices = ", ".join(["none", *sorted(VALIDATORS)])
        raise ValueError(f"未知验证器 {name!r}，可选：{choices}") from exc
    return validator_cls()


def validate_turn(
    validator: Validator | None,
    before: Observation,
    decision: PolicyDecision,
    after: Observation,
) -> TurnValidation | None:
    if validator is None:
        return None

    print("验证中...")
    result = validator.validate(before, decision, after)
    status = "通过" if result.passed else "未通过"
    print(f"验证结果: {status} - {result.summary}")
    if result.evidence:
        print(f"验证证据: {result.evidence}")
    return TurnValidation(
        validator_name=validator.name,
        passed=result.passed,
        summary=result.summary,
        evidence=result.evidence,
    )


def run_once(prompt: str, policy: Policy, validator: Validator | None) -> None:
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
        after_observation = Observation(png_bytes=after_bytes, source="live")
        after_path = SCREENSHOT.parent / "screenshot_after.png"
        after_path.write_bytes(after_bytes)
        print(f"已保存: {after_path}")
        validate_turn(validator, observation, decision, after_observation)


def run_continue(
    prompt: str,
    policy: Policy,
    validator: Validator | None,
    context_path: Path,
) -> None:
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
            validation = None
            if executed:
                time.sleep(1.5)
                print("动作后截图...")
                after_bytes = phone.screenshot()
                after_observation = Observation(png_bytes=after_bytes, source="live")
                after_path = SCREENSHOT.parent / f"screenshot_after_turn_{turn_no}.png"
                after_path.write_bytes(after_bytes)
                print(f"已保存: {after_path}")
                validation = validate_turn(validator, observation, decision, after_observation)

            policy.append_turn(context, observation, decision, executed, validation)
            policy.save_context(context_path, context)
            print(f"Context 已保存: {context_path}")
            if not executed:
                return

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
    parser.add_argument(
        "--validator",
        default=SimpleLLMValidator.name,
        choices=["none", *sorted(VALIDATORS)],
        help="动作后验证器",
    )
    args = parser.parse_args()

    policy = build_policy(args.policy)
    validator = build_validator(args.validator)
    if args.continue_path:
        run_continue(args.prompt, policy, validator, args.continue_path)
    else:
        run_once(args.prompt, policy, validator)


if __name__ == "__main__":
    main()

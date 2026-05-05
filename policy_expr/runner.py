"""CLI runner for single-turn policy experiments."""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from policy_expr.executor import ActionExecutor
from policy_expr.output import render_final_output
from policy_expr.perception import LivePerception, LivePhoneSession
from policy_expr.policies import StructuredOutputPolicy
from policy_expr.policies.base import Policy
from policy_expr.schemas import (
    Observation,
    PolicyContext,
    PolicyDecision,
    PolicyTurn,
    TurnValidation,
)
from policy_expr.validators import SimpleLLMValidator, Validator
from policy_expr.visualize import print_decision

POLICIES: dict[str, type[Policy]] = {
    StructuredOutputPolicy.name: StructuredOutputPolicy,
}
VALIDATORS: dict[str, type[Validator]] = {
    SimpleLLMValidator.name: SimpleLLMValidator,
}

ROOT = Path(__file__).parent.parent
POLICY_LOG_ROOT = ROOT / "logs" / "policy_expr"


def create_run_dir(mode: str) -> Path:
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = POLICY_LOG_ROOT / mode / started_at
    suffix = 2
    while path.exists():
        path = POLICY_LOG_ROOT / mode / f"{started_at}_{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    goal: str = "",
) -> TurnValidation | None:
    if validator is None:
        return None

    print("验证中...")
    result = validator.validate(before, decision, after, goal=goal)
    status = "通过" if result.passed else "未通过"
    print(f"验证结果: {status} - {result.summary}")
    if result.evidence:
        print(f"验证证据: {result.evidence}")
    if result.goal_completed is not None:
        completed = "已达成" if result.goal_completed else "未达成"
        print(f"目标状态: {completed} - {result.goal_completed_reason}")
    return TurnValidation(
        validator_name=validator.name,
        passed=result.passed,
        summary=result.summary,
        evidence=result.evidence,
        goal_completed=result.goal_completed,
        goal_completed_reason=result.goal_completed_reason,
    )


def emit_final_output(
    goal: str,
    policy_name: str,
    turns: list[PolicyTurn],
    log_dir: Path,
    stop_reason: str,
) -> str:
    output = render_final_output(goal, policy_name, turns, log_dir, stop_reason)
    print("\n" + "=" * 50)
    print("最终输出")
    print("=" * 50)
    print(output.rstrip())
    print("=" * 50)
    return output


def run_once(
    prompt: str,
    policy: Policy,
    validator: Validator | None,
    log_dir: Path,
    context_path: Path,
) -> None:
    context = PolicyContext(goal=prompt, policy_name=policy.name)
    policy.save_context(context_path, context)
    print(f"Goal    : {context.goal}")
    print(f"Turns   : {len(context.turns)}")

    with LivePhoneSession() as phone:
        perception = LivePerception(phone, log_dir / "screenshot.png")
        observation = perception.observe()

        print("分析中...")
        decision = policy.decide(observation, prompt)
        print_decision(decision, observation.png_bytes, log_dir / "structured_output_result.png")

        executed = ActionExecutor(phone).execute(decision)
        if not executed:
            policy.append_turn(context, observation, decision, executed)
            context.output = emit_final_output(
                context.goal,
                context.policy_name,
                context.turns,
                log_dir,
                "动作未执行，运行停止",
            )
            policy.save_context(context_path, context)
            print(f"Context 已保存: {context_path}")
            return

        time.sleep(1.5)
        print("点击后截图...")
        after_bytes = phone.screenshot()
        after_observation = Observation(png_bytes=after_bytes, source="live")
        after_path = log_dir / "screenshot_after.png"
        after_path.write_bytes(after_bytes)
        print(f"已保存: {after_path}")
        validation = validate_turn(validator, observation, decision, after_observation, goal=context.goal)
        policy.append_turn(context, observation, decision, executed, validation)
        context.output = emit_final_output(
            context.goal,
            context.policy_name,
            context.turns,
            log_dir,
            "single-step 完成一轮后停止",
        )
        policy.save_context(context_path, context)
        print(f"Context 已保存: {context_path}")


def run_agent_loop(
    prompt: str,
    policy: Policy,
    validator: Validator | None,
    input_context_path: Path | None,
    log_dir: Path,
    context_path: Path,
) -> None:
    if input_context_path is None:
        context = PolicyContext(goal=prompt, policy_name=policy.name)
    else:
        context = policy.load_context(input_context_path, prompt)
    policy.save_context(context_path, context)
    print(f"Goal    : {context.goal}")
    print(f"Turns   : {len(context.turns)}")

    with LivePhoneSession() as phone:
        executor = ActionExecutor(phone)

        while True:
            turn_no = len(context.turns) + 1
            print(f"\n--- Turn {turn_no} ---")
            perception = LivePerception(phone, log_dir / f"screenshot_turn_{turn_no}.png")
            observation = perception.observe()

            print("分析中...")
            decision = policy.decide_with_context(observation, context)
            print_decision(decision, observation.png_bytes, log_dir / f"structured_output_result_turn_{turn_no}.png")

            executed = executor.execute(decision)
            validation = None
            if executed:
                time.sleep(1.5)
                print("动作后截图...")
                after_bytes = phone.screenshot()
                after_observation = Observation(png_bytes=after_bytes, source="live")
                after_path = log_dir / f"screenshot_after_turn_{turn_no}.png"
                after_path.write_bytes(after_bytes)
                print(f"已保存: {after_path}")
                validation = validate_turn(validator, observation, decision, after_observation, goal=context.goal)

            policy.append_turn(context, observation, decision, executed, validation)
            if not executed:
                context.output = emit_final_output(
                    context.goal,
                    context.policy_name,
                    context.turns,
                    log_dir,
                    "动作未执行，agent-loop 停止",
                )
                policy.save_context(context_path, context)
                print(f"Context 已保存: {context_path}")
                return

            if validation and validation.goal_completed:
                print(f"\n目标已达成：{validation.goal_completed_reason}")
                context.output = emit_final_output(
                    context.goal,
                    context.policy_name,
                    context.turns,
                    log_dir,
                    f"目标已达成：{validation.goal_completed_reason}",
                )
                policy.save_context(context_path, context)
                print(f"Context 已保存: {context_path}")
                return

            policy.save_context(context_path, context)
            print(f"Context 已保存: {context_path}")
            answer = input("继续下一轮 ReAct？[Enter继续 / q退出] ").strip().lower()
            if answer in {"q", "quit", "exit"}:
                context.output = emit_final_output(
                    context.goal,
                    context.policy_name,
                    context.turns,
                    log_dir,
                    "用户退出 agent-loop",
                )
                policy.save_context(context_path, context)
                print(f"Context 已保存: {context_path}")
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="测试手机策略运行模式")
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
        "--mode",
        default="single-step",
        choices=["single-step", "agent-loop"],
        help="运行模式：single-step 单步 ReAct；agent-loop 单目标自动多步 ReAct",
    )
    parser.add_argument(
        "--context",
        type=Path,
        help="agent-loop 可选的 context 加载路径；本次运行的 context 固定保存到 logs/policy_expr/<mode>/<启动时间>/context.json",
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
    mode = args.mode
    input_context_path = args.context
    log_dir = create_run_dir(mode)
    context_path = log_dir / "context.json"
    print(f"Log Dir : {log_dir}")
    print(f"Context : {input_context_path if input_context_path else None}")

    if mode == "single-step":
        if input_context_path is not None:
            raise ValueError("--context 目前只支持 agent-loop 模式")
        run_once(args.prompt, policy, validator, log_dir, context_path)
    elif mode == "agent-loop":
        run_agent_loop(args.prompt, policy, validator, input_context_path, log_dir, context_path)


if __name__ == "__main__":
    main()

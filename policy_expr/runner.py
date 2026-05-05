"""CLI runner for single-turn policy experiments."""

import argparse
import json
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
    DialogContext,
    DialogMessage,
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


def load_dialog_context(path: Path, policy_name: str) -> DialogContext:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return DialogContext.model_validate(data)
    return DialogContext(policy_name=policy_name)


def save_dialog_context(path: Path, context: DialogContext) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(context.model_dump_json(indent=2), encoding="utf-8")


def append_dialog_turn(
    context: DialogContext,
    observation: Observation,
    decision: PolicyDecision,
    executed: bool,
    validation: TurnValidation | None,
) -> None:
    context.turns.append(
        PolicyTurn(
            index=len(context.turns) + 1,
            observation_source=observation.source,
            screen_type=decision.screen_type,
            app_name=decision.app_name,
            summary=decision.summary,
            reasoning=decision.reasoning,
            action=decision.action,
            executed=executed,
            validation=validation,
        )
    )


def build_dialog_prompt(context: DialogContext, user_prompt: str) -> str:
    recent_messages = context.messages[-8:]
    message_history = "\n".join(f"{m.role}: {m.content}" for m in recent_messages) or "无"

    def _format_turn(turn: PolicyTurn) -> str:
        return f"{turn.index}. {turn.summary}；动作：[{turn.action.action_type}] {turn.action.description}；执行：{turn.executed}"

    turn_history = "\n".join(_format_turn(t) for t in context.turns[-6:]) or "无"
    return (
        f"当前用户指令：{user_prompt}\n\n"
        "这是单用户多轮自然语言对话模式。请结合对话历史、手机操作历史和当前截图，"
        "只输出当前这一轮最应该执行的一个动作。\n\n"
        f"对话历史：\n{message_history}\n\n"
        f"手机操作历史：\n{turn_history}"
    )


def run_dialog_loop(
    policy: Policy,
    validator: Validator | None,
    input_context_path: Path | None,
    log_dir: Path,
    context_path: Path,
) -> None:
    _ = policy, validator
    context = load_dialog_context(input_context_path, policy.name) if input_context_path else DialogContext(policy_name=policy.name)
    print("TODO: dialog-loop 框架已保留，但单用户多轮自然语言对话模式尚未完整实现。")
    print("当前可使用 single-step --context 测试带历史的一步决策。")
    context.output = emit_final_output(
        "未提供",
        context.policy_name,
        context.turns,
        log_dir,
        "dialog-loop 尚未实现，未执行策略回合",
    )
    save_dialog_context(context_path, context)
    print(f"Context 已保存: {context_path}")


def run_single_step_with_context(
    prompt: str,
    policy: Policy,
    validator: Validator | None,
    input_context_path: Path | None,
    log_dir: Path,
    context_path: Path,
) -> None:
    context = load_dialog_context(input_context_path, policy.name) if input_context_path else DialogContext(policy_name=policy.name)
    context.messages.append(DialogMessage(role="user", content=prompt))
    save_dialog_context(context_path, context)
    print(f"Messages: {len(context.messages)}")
    print(f"Turns   : {len(context.turns)}")

    with LivePhoneSession() as phone:
        turn_no = len(context.turns) + 1
        perception = LivePerception(phone, log_dir / f"context_step_screenshot_turn_{turn_no}.png")
        observation = perception.observe()
        policy_prompt = build_dialog_prompt(context, prompt)

        print("分析中...")
        decision = policy.decide(observation, policy_prompt)
        print_decision(decision, observation.png_bytes, log_dir / f"context_step_result_turn_{turn_no}.png")

        executed = ActionExecutor(phone).execute(decision)
        validation = None
        if executed:
            time.sleep(1.5)
            print("动作后截图...")
            after_bytes = phone.screenshot()
            after_observation = Observation(png_bytes=after_bytes, source="live")
            after_path = log_dir / f"context_step_after_turn_{turn_no}.png"
            after_path.write_bytes(after_bytes)
            print(f"已保存: {after_path}")
            validation = validate_turn(validator, observation, decision, after_observation, goal=prompt)

        append_dialog_turn(context, observation, decision, executed, validation)
        context.messages.append(DialogMessage(role="assistant", content=f"{decision.summary}；{decision.action.description}"))
        context.output = emit_final_output(
            prompt,
            context.policy_name,
            context.turns,
            log_dir,
            "带上下文 single-step 完成一轮后停止",
        )
        save_dialog_context(context_path, context)
        print(f"Context 已保存: {context_path}")


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
        choices=["single-step", "agent-loop", "dialog-loop"],
        help="运行模式：single-step 单步 ReAct；agent-loop 单目标自动多步 ReAct；dialog-loop 单用户多轮自然语言对话",
    )
    parser.add_argument(
        "--context",
        type=Path,
        help="可选的 context 加载路径；本次运行的 context 固定保存到 logs/policy_expr/<mode>/<启动时间>/context.json",
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
        if input_context_path is None:
            run_once(args.prompt, policy, validator, log_dir, context_path)
        else:
            run_single_step_with_context(args.prompt, policy, validator, input_context_path, log_dir, context_path)
    elif mode == "agent-loop":
        run_agent_loop(args.prompt, policy, validator, input_context_path, log_dir, context_path)
    elif mode == "dialog-loop":
        run_dialog_loop(policy, validator, input_context_path, log_dir, context_path)


if __name__ == "__main__":
    main()

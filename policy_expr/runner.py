"""CLI runner for policy experiments with two-layer architecture."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from policy_expr.executor import ActionExecutor
from policy_expr.supervisor import MilestoneSupervisorPolicy, SimpleSupervisorPolicy
from policy_expr.supervisor.base import SupervisorPolicy
from policy_expr.output import render_final_output
from policy_expr.perception import LivePerception, LivePhoneSession
from policy_expr.policies import StructuredOutputPolicy
from policy_expr.policies.base import ActionPolicy
from policy_expr.schemas import (
    ActionDecision,
    Observation,
    PolicyContext,
    PolicyTurn,
    SupervisorStep,
)
from policy_expr.visualize import print_decision

POLICIES: dict[str, type[ActionPolicy]] = {
    StructuredOutputPolicy.name: StructuredOutputPolicy,
}

SUPERVISORS: dict[str, type[SupervisorPolicy]] = {
    SimpleSupervisorPolicy.name: SimpleSupervisorPolicy,
    MilestoneSupervisorPolicy.name: MilestoneSupervisorPolicy,
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


def build_policy(name: str) -> ActionPolicy:
    try:
        return POLICIES[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(POLICIES))
        raise ValueError(f"未知策略 {name!r}，可选：{choices}") from exc


def build_supervisor(name: str) -> SupervisorPolicy:
    try:
        return SUPERVISORS[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(SUPERVISORS))
        raise ValueError(f"未知监督者 {name!r}，可选：{choices}") from exc


def _save_context(path: Path, context: PolicyContext) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(context.model_dump_json(indent=2), encoding="utf-8")


def _load_context(
    path: Path,
    prompt: str,
    supervisor_name: str,
    action_name: str,
) -> PolicyContext:
    if path.exists():
        return PolicyContext.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return PolicyContext(
        goal=prompt,
        supervisor_policy_name=supervisor_name,
        action_policy_name=action_name,
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
    action_policy: ActionPolicy,
    supervisor: SupervisorPolicy,
    log_dir: Path,
    context_path: Path,
) -> None:
    context = PolicyContext(
        goal=prompt,
        supervisor_policy_name=supervisor.name,
        action_policy_name=action_policy.name,
    )
    _save_context(context_path, context)
    print(f"Goal    : {context.goal}")
    print(f"Turns   : {len(context.turns)}")

    with LivePhoneSession() as phone:
        perception = LivePerception(phone, log_dir / "screenshot.png")
        observation = perception.observe()

        print("监督决策中...")
        sv_step = supervisor.step(observation, context.goal, context.turns)
        print(f"监督者: {sv_step.summary}")

        action_decision = None
        executed = False

        if sv_step.should_act:
            print(f"动作指令: {sv_step.instruction}")
            print("动作决策中...")
            action_decision = action_policy.decide(observation, sv_step.instruction)
            print_decision(action_decision, observation.png_bytes, log_dir / "structured_output_result.png")
            executed = ActionExecutor(phone).execute(action_decision, app_name=sv_step.app_name or "")

        turn = PolicyTurn(
            index=1,
            observation_source=observation.source,
            supervisor=SupervisorStep(
                should_act=sv_step.should_act,
                instruction=sv_step.instruction,
                stop=sv_step.stop,
                stop_reason=sv_step.stop_reason,
                goal_completed=sv_step.goal_completed,
                summary=sv_step.summary,
            ),
            action_decision=action_decision,
            executed=executed,
        )
        context.turns.append(turn)
        _save_context(context_path, context)

        if executed:
            time.sleep(1.5)
            after_bytes = phone.screenshot()
            after_obs = Observation(png_bytes=after_bytes, source="live")
            after_path = log_dir / "screenshot_after.png"
            after_path.write_bytes(after_bytes)
            print(f"已保存: {after_path}")

            print("验证中...")
            confirm = supervisor.step(after_obs, context.goal, context.turns)
            context.turns.append(PolicyTurn(
                index=2,
                observation_source="live",
                supervisor=SupervisorStep(
                    should_act=confirm.should_act,
                    instruction=confirm.instruction,
                    stop=confirm.stop,
                    stop_reason=confirm.stop_reason,
                    goal_completed=confirm.goal_completed,
                    summary=confirm.summary,
                ),
                action_decision=None,
                executed=False,
            ))
            stop_reason = confirm.stop_reason or "single-step 完成一轮后停止"
        else:
            stop_reason = "动作未执行，single-step 停止"

        # context.output = emit_final_output(
        #     context.goal,
        #     supervisor.name,
        #     context.turns,
        #     log_dir,
        #     stop_reason,
        # )
        _save_context(context_path, context)
        print(f"Context 已保存: {context_path}")


def run_agent_loop(
    prompt: str,
    action_policy: ActionPolicy,
    supervisor: SupervisorPolicy,
    input_context_path: Path | None,
    log_dir: Path,
    context_path: Path,
) -> None:
    context = _load_context(
        input_context_path or context_path,
        prompt,
        supervisor.name,
        action_policy.name,
    )
    _save_context(context_path, context)
    print(f"Goal    : {context.goal}")
    print(f"Turns   : {len(context.turns)}")

    with LivePhoneSession() as phone:
        executor = ActionExecutor(phone)

        while True:
            turn_no = len(context.turns) + 1
            print(f"\n--- Turn {turn_no} ---")

            perception = LivePerception(phone, log_dir / f"screenshot_turn_{turn_no}.png")
            observation = perception.observe()

            print("监督决策中...")
            sv_step = supervisor.step(observation, context.goal, context.turns)
            print(f"监督者: {sv_step.summary}")

            action_decision = None
            executed = False

            if sv_step.should_act:
                print(f"动作指令: {sv_step.instruction}")
                print("动作决策中...")
                action_decision = action_policy.decide(observation, sv_step.instruction)
                print_decision(
                    action_decision,
                    observation.png_bytes,
                    log_dir / f"structured_output_result_turn_{turn_no}.png",
                )
                executed = executor.execute(action_decision, app_name=sv_step.app_name or "")

            turn = PolicyTurn(
                index=turn_no,
                observation_source=observation.source,
                supervisor=SupervisorStep(
                    should_act=sv_step.should_act,
                    instruction=sv_step.instruction,
                    stop=sv_step.stop,
                    stop_reason=sv_step.stop_reason,
                    goal_completed=sv_step.goal_completed,
                    summary=sv_step.summary,
                ),
                action_decision=action_decision,
                executed=executed,
            )
            context.turns.append(turn)
            _save_context(context_path, context)
            print(f"Context 已保存: {context_path}")

            if sv_step.stop or sv_step.goal_completed:
                reason = sv_step.stop_reason or "目标已达成"
                print(f"\n目标已达成：{reason}")
                # context.output = emit_final_output(
                #     context.goal,
                #     supervisor.name,
                #     context.turns,
                #     log_dir,
                #     reason,
                # )
                _save_context(context_path, context)
                return

            if not executed and sv_step.should_act:
                # context.output = emit_final_output(
                #     context.goal,
                #     supervisor.name,
                #     context.turns,
                #     log_dir,
                #     "动作未执行，agent-loop 停止",
                # )
                _save_context(context_path, context)
                return

            answer = input("继续下一轮？[Enter继续 / q退出] ").strip().lower()
            if answer in {"q", "quit", "exit"}:
                # context.output = emit_final_output(
                #     context.goal,
                #     supervisor.name,
                #     context.turns,
                #     log_dir,
                #     "用户退出 agent-loop",
                # )
                _save_context(context_path, context)
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
        help="动作策略模块",
    )
    parser.add_argument(
        "--supervisor",
        default=SimpleSupervisorPolicy.name,
        choices=sorted(SUPERVISORS),
        help="监督者策略模块",
    )
    parser.add_argument(
        "--mode",
        default="single-step",
        choices=["single-step", "agent-loop"],
        help="运行模式：single-step 单步；agent-loop 多步自动循环",
    )
    parser.add_argument(
        "--context",
        type=Path,
        help="agent-loop 可选的 context 加载路径",
    )
    args = parser.parse_args()

    action_policy = build_policy(args.policy)
    supervisor = build_supervisor(args.supervisor)
    mode = args.mode
    input_context_path = args.context
    log_dir = create_run_dir(mode)
    context_path = log_dir / "context.json"
    print(f"Log Dir : {log_dir}")
    print(f"Context : {input_context_path if input_context_path else None}")

    if mode == "single-step":
        if input_context_path is not None:
            raise ValueError("--context 目前只支持 agent-loop 模式")
        run_once(args.prompt, action_policy, supervisor, log_dir, context_path)
    elif mode == "agent-loop":
        run_agent_loop(args.prompt, action_policy, supervisor, input_context_path, log_dir, context_path)


if __name__ == "__main__":
    main()

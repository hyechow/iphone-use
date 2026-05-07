"""CLI runner for policy experiments with two-layer architecture."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.structured import get_llm_call_count
from policy_expr.executor import ActionExecutor, mean_image_diff
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
TURN_HEADER = "\033[1;36m--- Turn {turn_no} ---\033[0m"
TURN_STATS = "\033[2mTurn {turn_no} stats: llm_calls={llm_calls}, elapsed={elapsed:.2f}s\033[0m"

MAX_ACTION_RETRIES = 2        # 动作无效时最多重试次数
ACTION_EFFECT_THRESHOLD = 3.0  # mean_image_diff 低于此值视为动作未生效


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


def _print_turn_stats(turn_no: int, started_at: float, llm_calls_before: int) -> None:
    elapsed = time.perf_counter() - started_at
    llm_calls = get_llm_call_count() - llm_calls_before
    print(TURN_STATS.format(turn_no=turn_no, llm_calls=llm_calls, elapsed=elapsed))


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
        turn_started_at = time.perf_counter()
        llm_calls_before = get_llm_call_count()

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
        _print_turn_stats(1, turn_started_at, llm_calls_before)


def run_agent_loop(
    prompt: str,
    action_policy: ActionPolicy,
    supervisor: SupervisorPolicy,
    input_context_path: Path | None,
    log_dir: Path,
    context_path: Path,
    max_turns: int = 20,
    auto_continue: bool = False,
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
            if turn_no > max_turns:
                print(f"\n达到最大轮数 {max_turns}，agent-loop 停止")
                _save_context(context_path, context)
                return

            turn_started_at = time.perf_counter()
            llm_calls_before = get_llm_call_count()

            print("\n" + TURN_HEADER.format(turn_no=turn_no))

            perception = LivePerception(phone, log_dir / f"screenshot_turn_{turn_no}.png")
            observation = perception.observe()

            print("监督决策中...")
            sv_step = supervisor.step(observation, context.goal, context.turns)
            print(f"监督者: {sv_step.summary}")

            action_decision = None
            executed = False

            if sv_step.should_act:
                print(f"动作指令: {sv_step.instruction}")
                action_obs = observation
                instruction = sv_step.instruction
                action_decision = None
                executed = False

                for attempt in range(MAX_ACTION_RETRIES + 1):
                    if attempt > 0:
                        # 取最新截图，附加未命中提示
                        new_png = phone.screenshot()
                        action_obs = Observation(png_bytes=new_png, source="live")
                        instruction = (
                            f"{sv_step.instruction}\n\n"
                            "注意：上次点击可能未命中目标，请仔细核对截图中的元素位置后重新确定坐标。"
                        )
                        print(f"  [重试 {attempt}/{MAX_ACTION_RETRIES}] 重新决策动作...")

                    print("动作决策中...")
                    action_decision = action_policy.decide(action_obs, instruction)
                    print_decision(
                        action_decision,
                        action_obs.png_bytes,
                        log_dir / f"structured_output_result_turn_{turn_no}.png",
                    )
                    executed = executor.execute(action_decision, app_name=sv_step.app_name or "")

                    if not executed or attempt >= MAX_ACTION_RETRIES:
                        break

                    # 等 UI 响应后检查动作是否生效
                    time.sleep(0.8)
                    post_png = phone.screenshot()
                    diff = mean_image_diff(action_obs.png_bytes, post_png)
                    print(f"  [效果检测] mean_diff={diff:.2f}", end="")
                    if diff >= ACTION_EFFECT_THRESHOLD:
                        print(" → 生效")
                        break
                    print(f" → 未生效（< {ACTION_EFFECT_THRESHOLD}），重试")

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
            _print_turn_stats(turn_no, turn_started_at, llm_calls_before)

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

            if not sv_step.should_act:
                # milestone 完成 / 未执行动作 → 自动继续下一轮，不暂停
                continue

            if auto_continue:
                time.sleep(1.5)
                continue

            try:
                answer = input("继续下一轮？[Enter继续 / q退出] ").strip().lower()
            except EOFError:
                answer = ""
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
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="agent-loop 最大自动执行轮数，防止无限循环",
    )
    parser.add_argument(
        "--auto-continue",
        action="store_true",
        help="agent-loop 动作执行后自动进入下一轮；默认手动确认",
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
        run_agent_loop(
            args.prompt,
            action_policy,
            supervisor,
            input_context_path,
            log_dir,
            context_path,
            max_turns=args.max_turns,
            auto_continue=args.auto_continue,
        )


if __name__ == "__main__":
    main()

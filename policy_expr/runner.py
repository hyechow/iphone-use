"""CLI runner for policy experiments with two-layer architecture."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from llm.structured import get_llm_call_count
from policy_expr.executor import ActionExecutor
from policy_expr.supervisor import MilestoneSupervisorPolicy, SimpleSupervisorPolicy
from policy_expr.supervisor.base import SupervisorPolicy
from policy_expr.output import render_final_output
from policy_expr.perception import LivePerception, LivePhoneSession
from policy_expr.reader import ContentReader
from policy_expr.policies import StructuredOutputPolicy
from policy_expr.policies.base import ActionPolicy
from policy_expr.schemas import (
    ActionDecision,
    GoalValidationResult,
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

# 动作重试机制暂时关闭：每轮只做一次 action policy 决策和执行。
# MAX_ACTION_RETRIES = 2        # 动作无效时最多重试次数
# ACTION_EFFECT_THRESHOLD = 3.0  # mean_image_diff 低于此值视为动作未生效


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


VALIDATION_PROMPT = """\
判断以下收集到的数据片段是否充分回答了用户目标。

用户目标：{goal}

收集到的数据：
{notes_text}

要求：
- 如果目标包含特定条件（如时间范围、金额范围、类别），检查数据是否满足这些条件
- 如果数据范围与目标条件不匹配（如目标问"本周"但数据是"本月"），判定为不充分
- 如果数据只是部分满足，也判定为不充分
- 只有数据明确、完整地回答了用户目标时，才判定为充分

输出 JSON：
- sufficient: true/false
- missing: 数据缺少什么（sufficient=false 时必填，sufficient=true 时留空）
"""


def validate_goal_completion(goal: str, content_notes: list[str]) -> GoalValidationResult:
    """独立 LLM 校验：收集到的数据是否充分回答了用户目标。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from llm.structured import invoke_structured
    from policy_expr.config import resolve_llm_config

    cfg = resolve_llm_config("output")
    llm = ChatOpenAI(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)

    notes_text = "\n\n".join(f"[片段 {i+1}]\n{note}" for i, note in enumerate(content_notes))
    prompt = VALIDATION_PROMPT.format(goal=goal, notes_text=notes_text)

    messages = [
        SystemMessage(content="你是数据充分性校验助手。"),
        HumanMessage(content=prompt),
    ]
    return invoke_structured(llm, messages, GoalValidationResult)


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
    content_notes: list[str] | None = None,
) -> str:
    output = render_final_output(goal, policy_name, turns, log_dir, stop_reason, content_notes)
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

        context.output = emit_final_output(
            context.goal,
            supervisor.name,
            context.turns,
            log_dir,
            stop_reason,
            content_notes=context.content_notes or None,
        )
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

    reader = ContentReader()

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

            # Sync task_type from supervisor after first decomposition
            if hasattr(supervisor, "task_type") and context.task_type is None:
                context.task_type = supervisor.task_type
                print(f"任务类型: {context.task_type}")

            # Content reading for analysis tasks
            if sv_step.read_instruction:
                print(f"读取内容: {sv_step.read_instruction}")
                note = reader.read(observation.png_bytes, sv_step.read_instruction)
                if note and note != "无相关内容":
                    context.content_notes.append(note)
                    print(f"内容摘要: {note[:80]}...")

            action_decision = None
            executed = False

            if sv_step.should_act:
                print(f"动作指令: {sv_step.instruction}")
                if sv_step.preformed_action:
                    print("使用预生成动作，跳过 Action Policy")
                    action_decision = sv_step.preformed_action
                else:
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
                llm_calls=get_llm_call_count() - llm_calls_before,
            )
            context.turns.append(turn)
            _save_context(context_path, context)
            _print_turn_stats(turn_no, turn_started_at, llm_calls_before)

            if sv_step.stop or sv_step.goal_completed:
                # analysis 任务且有 content_notes → 二次校验
                if context.task_type == "analysis" and context.content_notes:
                    print("  [校验] 数据充分性检查...")
                    validation = validate_goal_completion(context.goal, context.content_notes)
                    if not validation.sufficient:
                        print(f"  [校验] 数据不充分: {validation.missing}，继续执行")
                        # 注入校验反馈到 goal，让 supervisor 知道缺少什么
                        context.goal = (
                            f"{context.goal}\n\n"
                            f"[校验反馈：之前收集的数据不完整——{validation.missing}，"
                            f"请换一种方式获取所需数据。]"
                        )
                        # 回退本轮 turn 的完成标记
                        turn.supervisor.stop = False
                        turn.supervisor.goal_completed = False
                        _save_context(context_path, context)
                        continue

                reason = sv_step.stop_reason or "目标已达成"
                print(f"\n目标已达成：{reason}")
                context.output = emit_final_output(
                    context.goal,
                    supervisor.name,
                    context.turns,
                    log_dir,
                    reason,
                    content_notes=context.content_notes or None,
                )
                _save_context(context_path, context)
                return

            if not executed and sv_step.should_act:
                context.output = emit_final_output(
                    context.goal,
                    supervisor.name,
                    context.turns,
                    log_dir,
                    "动作未执行，agent-loop 停止",
                    content_notes=context.content_notes or None,
                )
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
                context.output = emit_final_output(
                    context.goal,
                    supervisor.name,
                    context.turns,
                    log_dir,
                    "用户退出 agent-loop",
                    content_notes=context.content_notes or None,
                )
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

"""CLI runner for policy experiments with two-layer architecture."""

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import IO, Iterator

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from llm.structured import get_llm_call_count
from policy_expr.executor import ActionExecutor
from policy_expr.supervisor import MilestoneSupervisorPolicy, SimpleSupervisorPolicy
from policy_expr.supervisor.base import SupervisorPolicy
from policy_expr.output import render_final_output, validate_goal_completion
from policy_expr.perception import LivePerception, LivePhoneSession
from policy_expr.reader import ContentReader, annotate_content_note, build_reader_instruction
from policy_expr.policies import StructuredOutputPolicy
from policy_expr.policies.base import ActionPolicy
from policy_expr.schemas import (
    Observation,
    PolicyContext,
    PolicyTurn,
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


class _TeeStream:
    """Write text to both the original stream and a log file."""

    def __init__(self, original: IO[str], log_file: IO[str]) -> None:
        self._original = original
        self._log_file = log_file
        self.encoding = getattr(original, "encoding", "utf-8")
        self.errors = getattr(original, "errors", "replace")

    def write(self, text: str) -> int:
        written = self._original.write(text)
        self._log_file.write(text)
        return written

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._original.isatty()

    def fileno(self) -> int:
        return self._original.fileno()

    def __getattr__(self, name: str) -> object:
        return getattr(self._original, name)


@contextmanager
def _tee_stdio(log_dir: Path) -> Iterator[None]:
    """Mirror stdout/stderr to per-run text logs while preserving terminal output."""

    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    with (
        stdout_path.open("a", encoding="utf-8", buffering=1) as stdout_file,
        stderr_path.open("a", encoding="utf-8", buffering=1) as stderr_file,
        redirect_stdout(_TeeStream(sys.stdout, stdout_file)),
        redirect_stderr(_TeeStream(sys.stderr, stderr_file)),
    ):
        print(f"Stdout  : {stdout_path}")
        print(f"Stderr  : {stderr_path}")
        try:
            yield
        except Exception:
            traceback.print_exc()
            raise SystemExit(1) from None


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


def _ensure_note_hashes(context: PolicyContext) -> None:
    if context.content_notes and not context.content_note_hashes:
        context.content_note_hashes = [_note_hash(note) for note in context.content_notes]


def _note_hash(note: str) -> str:
    normalized = re.sub(r"\s+", "", note.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()



def _supervisor_has_active_work(supervisor: SupervisorPolicy) -> bool:
    current_id = getattr(supervisor, "_current_id", None)
    if current_id is not None:
        return True
    return not isinstance(supervisor, MilestoneSupervisorPolicy)


def emit_final_output(
    goal: str,
    policy_name: str,
    turns: list[PolicyTurn],
    log_dir: Path,
    stop_reason: str,
    content_notes: list[str] | None = None,
    collection_context: str | None = None,
) -> str:
    output = render_final_output(goal, policy_name, turns, log_dir, stop_reason, content_notes, collection_context)
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
            supervisor=sv_step,
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
                supervisor=confirm,
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
    _ensure_note_hashes(context)
    _save_context(context_path, context)
    print(f"Goal    : {context.goal}")
    print(f"Turns   : {len(context.turns)}")

    reader = ContentReader()
    original_goal = context.goal
    validation_retries = 0
    max_validation_retries = 2
    noop_count = 0
    prev_milestone_id: str | None = None

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
            if sv_step.collection_scope and sv_step.collection_scope != context.collection_scope:
                context.collection_scope = sv_step.collection_scope
                print(
                    "采集范围: "
                    + json.dumps(context.collection_scope.model_dump(exclude_none=True), ensure_ascii=False)
                )

            read_added_content = False
            read_note_hash = None

            # Content reading is controlled by the current milestone strategy.
            if sv_step.read_instruction and not sv_step.allow_read:
                print(
                    "跳过读取入库: 当前阶段不允许采集 "
                    f"({sv_step.milestone_kind}/{sv_step.completion_strategy})"
                )
            elif sv_step.read_instruction:
                reader_instruction = build_reader_instruction(original_goal, sv_step)
                print(f"读取内容: {reader_instruction}")
                note = reader.read(observation.png_bytes, reader_instruction)
                if note and note != "无相关内容":
                    note = annotate_content_note(
                        note,
                        turn_no=turn_no,
                        sv_step=sv_step,
                        collection_scope=context.collection_scope,
                    )
                    read_note_hash = _note_hash(note)
                    if read_note_hash not in context.content_note_hashes:
                        context.content_note_hashes.append(read_note_hash)
                        context.content_notes.append(note)
                        read_added_content = True
                        print(f"内容摘要: {note[:80]}...")
                    else:
                        print("内容摘要: 与已采集内容重复，未入库")

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
                supervisor=sv_step,
                action_decision=action_decision,
                executed=executed,
                llm_calls=get_llm_call_count() - llm_calls_before,
                read_added_content=read_added_content,
                read_note_hash=read_note_hash,
            )
            context.turns.append(turn)
            _save_context(context_path, context)
            _print_turn_stats(turn_no, turn_started_at, llm_calls_before)

            if sv_step.stop or sv_step.goal_completed:
                # analysis 任务达成且有 content_notes → 二次校验
                if (
                    sv_step.goal_completed
                    and
                    context.task_type == "analysis"
                    and context.content_notes
                    and validation_retries < max_validation_retries
                ):
                    print("  [校验] 数据充分性检查...")
                    validation = validate_goal_completion(
                        original_goal, context.content_notes,
                        collection_context=sv_step.collection_summary,
                    )
                    if not validation.sufficient:
                        validation_retries += 1
                        if not _supervisor_has_active_work(supervisor):
                            reason = f"数据校验不充分：{validation.missing}"
                            print(
                                f"  [校验] 数据不充分 "
                                f"({validation_retries}/{max_validation_retries}): "
                                f"{validation.missing}，无可继续执行的子目标"
                            )
                            turn.supervisor.stop = True
                            turn.supervisor.goal_completed = False
                            _save_context(context_path, context)
                            print(f"\n任务未完成：{reason}")
                            context.output = emit_final_output(
                                original_goal,
                                supervisor.name,
                                context.turns,
                                log_dir,
                                reason,
                                content_notes=context.content_notes or None,
                                collection_context=sv_step.collection_summary,
                            )
                            _save_context(context_path, context)
                            return
                        print(
                            f"  [校验] 数据不充分 "
                            f"({validation_retries}/{max_validation_retries}): "
                            f"{validation.missing}，继续执行"
                        )
                        # 注入校验反馈到 goal（只追加一次，不重复）
                        context.goal = (
                            f"{original_goal}\n\n"
                            f"[校验反馈：之前收集的数据不完整——{validation.missing}，"
                            f"请换一种方式获取所需数据。]"
                        )
                        # 回退本轮 turn 的完成标记
                        turn.supervisor.stop = False
                        turn.supervisor.goal_completed = False
                        _save_context(context_path, context)
                        continue

                reason = sv_step.stop_reason or ("目标已达成" if sv_step.goal_completed else "agent-loop 停止")
                if sv_step.goal_completed:
                    print(f"\n目标已达成：{reason}")
                else:
                    print(f"\n任务未完成：{reason}")
                context.output = emit_final_output(
                    original_goal,
                    supervisor.name,
                    context.turns,
                    log_dir,
                    reason,
                    content_notes=context.content_notes or None,
                    collection_context=sv_step.collection_summary,
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

            if sv_step.milestone_id != prev_milestone_id:
                noop_count = 0
            prev_milestone_id = sv_step.milestone_id

            if not sv_step.should_act:
                noop_count += 1
                if noop_count >= 3:
                    print(f"\n连续 {noop_count} 轮无动作，agent-loop 停止")
                    context.output = emit_final_output(
                        original_goal,
                        supervisor.name,
                        context.turns,
                        log_dir,
                        f"连续 {noop_count} 轮无动作",
                        content_notes=context.content_notes or None,
                    )
                    _save_context(context_path, context)
                    return
                continue

            noop_count = 0  # 有动作则重置计数

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
    with _tee_stdio(log_dir):
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

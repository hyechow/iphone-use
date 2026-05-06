"""Milestone-based supervisor: decompose goal → track milestones → replan on failure."""

import base64
import json
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm.structured import invoke_structured
from policy_expr.config import resolve_llm_config
from policy_expr.policies.base import resize_to_logical_png
from policy_expr.schemas import Milestone, Observation, PolicyTurn, SupervisorStep

load_dotenv()

MAX_RETRIES = 3

# ── LLM response schemas ──────────────────────────────────────────────


class _DecomposeResponse(BaseModel):
    goal: str
    global_constraints: list[str] = Field(default_factory=list)
    milestones: list[Milestone]


class _CheckResult(BaseModel):
    """Checker output: evaluate milestone progress against success_condition."""
    status: str = Field(description="in_progress | done | stuck")
    reason: str = Field(description="判断理由")
    stuck_reason: str = Field(default="", description="卡住原因（仅 stuck 时填写）")
    issues: list[str] = Field(default_factory=list, description="具体问题列表")
    summary: str = Field(description="当前屏幕状态一句话描述")


class _PlanResult(BaseModel):
    """Planner output: next instruction for in_progress milestone."""
    instruction: str = Field(description="下一步精确操作指令")
    summary: str = Field(description="当前屏幕状态一句话描述")


class _ReplanResult(BaseModel):
    """Replanner output: diagnose failure and generate fix."""
    diagnosis: str = Field(description="失败根本原因（一句话）")
    strategy: str = Field(description="local_replan | escalate_human")
    instruction: str = Field(default="", description="修复操作指令（local_replan 时必填）")
    escalation_message: str = Field(default="", description="升级人工消息（escalate_human 时填写）")


# ── Prompts ───────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = """\
你是 iPhone 自动化任务的规划 Supervisor。将用户任务分解为子目标（milestone）。

可用操作：tap（点击）、type（输入）、scroll（滚动）、home（返回主屏幕）

输出要求：
- goal：任务一句话描述
- global_constraints：全局约束列表
- milestones：子目标列表，每个包含 id/name/description/depends_on/success_condition/failure_hints

原则：
1. 每个子目标 3-8 个操作内可完成
2. success_condition 要具体可判断，如「看到XX页面标题」「XX输入框有内容」
3. depends_on 填依赖的前置子目标 id，无依赖留空
4. 不指定具体 UI 坐标或元素位置（那是 action policy 的事）
5. 简单任务只需 1 个 milestone
6. failure_hints 列出该子目标可能失败的原因
"""

CHECKER_PROMPT = """\
你是 iPhone 自动化任务的验收员。根据当前屏幕截图和子目标验收条件，判断执行进展。

## 当前子目标
- 名称：{milestone_name}
- 描述：{milestone_desc}
- 验收条件：{success_condition}

## 历史操作记录
{history_text}

判断规则：
- done：屏幕显示符合验收条件
- stuck：出现错误弹窗、连续无进展、页面回退、与上一次截图无明显变化
- in_progress：正在向目标推进
- 不要凭感觉判断，只看可观测的事实
- issues 列出具体观察到的问题

输出 JSON：
- status: in_progress | done | stuck
- reason: 判断理由
- stuck_reason: 卡住原因（仅 stuck 时填写）
- issues: 具体问题列表
- summary: 当前屏幕状态一句话描述
"""

PLAN_PROMPT = """\
你是 iPhone 自动化任务的步骤规划器。根据当前子目标和历史操作记录，给出下一步操作指令。

## 当前子目标
- 名称：{milestone_name}
- 描述：{milestone_desc}
- 验收条件：{success_condition}
- 全局约束：{constraints}

## 历史操作记录
{history_text}

给出下一步操作指令：
- 描述要操作的目标元素，如「点击底部导航栏左起第二个通讯录图标」
- 不要给出目标级指令如「进入通讯录页面」
- 每次只给一个操作
"""

REPLAN_PROMPT = """\
你是 iPhone 自动化任务的修复规划器。某个子目标执行失败，请诊断原因并制定修复策略。

## 失败的子目标
- 名称：{milestone_name}
- 描述：{milestone_desc}
- 验收条件：{success_condition}
- 失败原因：{stuck_reason}
- 具体问题：{issues}
- 已重试次数：{retry_count}
- 全局约束：{constraints}
- 预期失败提示：{failure_hints}

## 历史操作记录
{history_text}

决策规则：
- 工具限制/数据问题 → local_replan，换一种操作方式
- 重试次数 >= 3 → escalate_human
- local_replan 的指令不能重复已失败的方案

输出 JSON：
- diagnosis: 失败根本原因（一句话）
- strategy: local_replan | escalate_human
- instruction: 修复操作指令（local_replan 时必填）
- escalation_message: 升级人工消息（escalate_human 时填写）
"""


# ── History formatter (reuse pattern from simple.py) ──────────────────


def _format_history(history: list[PolicyTurn]) -> str:
    if not history:
        return "（无历史记录，这是第一轮）"

    lines = []
    for turn in history[-8:]:
        sv = turn.supervisor
        if turn.action_decision and turn.executed:
            action = turn.action_decision.action
            lines.append(
                f"{turn.index}. [执行] [{action.action_type}] {action.description}"
                f"（{sv.summary}）"
            )
        elif turn.action_decision and not turn.executed:
            action = turn.action_decision.action
            lines.append(
                f"{turn.index}. [未执行] [{action.action_type}] {action.description}"
            )
        else:
            lines.append(f"{turn.index}. [跳过动作] {sv.summary}")
    return "\n".join(lines)


# ── Implementation ────────────────────────────────────────────────────


class MilestoneSupervisorPolicy:
    """Supervisor that decomposes goals into milestones and tracks progress.

    Flow:
    1. First step(): decompose → plan (no check, nothing executed yet)
    2. Subsequent steps: check → plan/replan
    3. Transitions: done → next milestone; stuck → Replanner (up to 3); failed → skip
    """

    name = "milestone"

    def __init__(self) -> None:
        self._global_constraints: list[str] = []
        self._milestones: dict[str, Milestone] = {}
        self._order: list[str] = []
        self._current_id: Optional[str] = None
        self._failure_reason: str = ""
        self._initialized = False

    def step(
        self,
        observation: Observation,
        goal: str,
        history: list[PolicyTurn],
    ) -> SupervisorStep:
        if not self._initialized:
            self._decompose(goal)
            self._initialized = True

        # No more milestones
        if self._current_id is None:
            return SupervisorStep(
                should_act=False,
                stop=True,
                stop_reason="所有子目标已完成",
                goal_completed=True,
                summary="任务完成",
            )

        milestone = self._milestones[self._current_id]

        # 首次 step（history 为空或 milestone 是 pending）：直接 plan，无需 check
        if milestone.status == "pending" or not history:
            milestone.status = "running"
            plan = self._plan(milestone, history)
            return SupervisorStep(
                should_act=True,
                instruction=plan.instruction,
                stop=False,
                goal_completed=False,
                summary=f"开始执行子目标「{milestone.name}」。{plan.summary}",
            )

        # ── Phase 1: Checker（有执行历史时才检查） ──
        check = self._check(milestone, observation, history)
        print(f"  [Checker] {check.status}: {check.reason}")

        if check.status == "done":
            milestone.status = "done"
            self._current_id = self._next_milestone()
            self._failure_reason = ""

            if self._current_id is None:
                return SupervisorStep(
                    should_act=False,
                    stop=True,
                    stop_reason="所有子目标已完成",
                    goal_completed=True,
                    summary=f"子目标「{milestone.name}」已完成，任务全部完成。",
                )
            return SupervisorStep(
                should_act=False,
                stop=False,
                goal_completed=False,
                summary=f"子目标「{milestone.name}」已完成，准备进入下一个。",
            )

        if check.status == "stuck":
            milestone.retry_count += 1
            self._failure_reason = check.stuck_reason or check.reason

            if milestone.retry_count >= MAX_RETRIES:
                milestone.status = "failed"
                self._current_id = self._next_milestone()
                self._failure_reason = ""

                if self._current_id is None:
                    return SupervisorStep(
                        should_act=False,
                        stop=True,
                        stop_reason=(
                            f"子目标「{milestone.name}」重试 {MAX_RETRIES} 次后失败"
                        ),
                        goal_completed=False,
                        summary=check.reason,
                    )
                return SupervisorStep(
                    should_act=False,
                    stop=False,
                    goal_completed=False,
                    summary=f"子目标「{milestone.name}」失败，跳过继续下一个。",
                )

            # ── Phase 2: Replanner ──
            print(f"  [Replanner] 第 {milestone.retry_count} 次重试...")
            replan = self._replan(milestone, check, observation, history)
            print(f"  [Replanner] 诊断: {replan.diagnosis}, 策略: {replan.strategy}")

            if replan.strategy == "escalate_human":
                milestone.status = "failed"
                self._current_id = self._next_milestone()
                self._failure_reason = ""
                return SupervisorStep(
                    should_act=False,
                    stop=self._current_id is None,
                    stop_reason=replan.escalation_message or "升级人工介入",
                    goal_completed=False,
                    summary=replan.diagnosis,
                )

            milestone.status = "running"
            return SupervisorStep(
                should_act=bool(replan.instruction),
                instruction=replan.instruction or None,
                stop=False,
                goal_completed=False,
                summary=(
                    f"子目标「{milestone.name}」卡住，"
                    f"第 {milestone.retry_count} 次重试。{replan.diagnosis}"
                ),
            )

        # in_progress ── Phase 2: Planner ──
        plan = self._plan(milestone, history)
        milestone.status = "running"
        return SupervisorStep(
            should_act=True,
            instruction=plan.instruction,
            stop=False,
            goal_completed=False,
            summary=plan.summary,
        )

    # ── Internal helpers ──────────────────────────────────────────────

    def _decompose(self, goal: str) -> None:
        cfg = resolve_llm_config("supervisor.checker")
        print(f"Checker Provider   : {cfg.provider}")
        print(f"Checker Model      : {cfg.model}")

        llm = ChatOpenAI(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)
        resp = invoke_structured(llm, [
            SystemMessage(content=DECOMPOSE_PROMPT),
            HumanMessage(content=f"用户任务：{goal}"),
        ], _DecomposeResponse)

        self._global_constraints = resp.global_constraints
        for m in resp.milestones:
            self._milestones[m.id] = m
        self._order = [m.id for m in resp.milestones]
        self._current_id = self._next_milestone()

        print(f"任务分解为 {len(self._milestones)} 个子目标：")
        for m in resp.milestones:
            deps = f" (依赖: {m.depends_on})" if m.depends_on else ""
            print(f"  [{m.id}] {m.name}{deps}")
            print(f"       验收：{m.success_condition}")

    def _next_milestone(self) -> Optional[str]:
        for mid in self._order:
            m = self._milestones[mid]
            if m.status != "pending":
                continue
            if all(
                self._milestones[dep].status == "done"
                for dep in m.depends_on
            ):
                return mid
        return None

    def _check(
        self, milestone: Milestone, observation: Observation, history: list[PolicyTurn],
    ) -> _CheckResult:
        """Checker: evaluate milestone progress against success_condition."""
        llm = self._make_llm()
        messages = self._build_messages(
            CHECKER_PROMPT.format(
                milestone_name=milestone.name,
                milestone_desc=milestone.description,
                success_condition=milestone.success_condition,
                history_text=_format_history(history),
            ),
            observation,
        )
        return invoke_structured(llm, messages, _CheckResult)

    def _plan(
        self, milestone: Milestone, history: list[PolicyTurn],
    ) -> _PlanResult:
        """Planner: generate next instruction (text-only, no screenshot)."""
        llm = self._make_planner_llm()
        user_text = PLAN_PROMPT.format(
            milestone_name=milestone.name,
            milestone_desc=milestone.description,
            success_condition=milestone.success_condition,
            constraints=json.dumps(self._global_constraints, ensure_ascii=False),
            history_text=_format_history(history),
        )
        messages = [
            SystemMessage(content=user_text),
            HumanMessage(content="请给出下一步操作指令。"),
        ]
        return invoke_structured(llm, messages, _PlanResult)

    def _replan(
        self, milestone: Milestone, check: _CheckResult,
        observation: Observation, history: list[PolicyTurn],
    ) -> _ReplanResult:
        """Replanner: diagnose failure and generate fix strategy."""
        llm = self._make_llm()
        messages = self._build_messages(
            REPLAN_PROMPT.format(
                milestone_name=milestone.name,
                milestone_desc=milestone.description,
                success_condition=milestone.success_condition,
                stuck_reason=check.stuck_reason or check.reason,
                issues=json.dumps(check.issues, ensure_ascii=False),
                retry_count=milestone.retry_count,
                constraints=json.dumps(self._global_constraints, ensure_ascii=False),
                failure_hints=json.dumps(milestone.failure_hints, ensure_ascii=False),
                history_text=_format_history(history),
            ),
            observation,
        )
        return invoke_structured(llm, messages, _ReplanResult)

    def _make_llm(self) -> ChatOpenAI:
        cfg = resolve_llm_config("supervisor.checker")
        return ChatOpenAI(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)

    def _make_planner_llm(self) -> ChatOpenAI:
        cfg = resolve_llm_config("supervisor.planner")
        return ChatOpenAI(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)

    def _build_messages(
        self, system_prompt: str, observation: Observation,
    ) -> list:
        """Build multimodal messages with system prompt + screenshot."""
        b64 = base64.b64encode(resize_to_logical_png(observation.png_bytes)).decode()
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": "请根据当前屏幕做出决策。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]),
        ]


# ── CLI entry point: uv run python -m policy_expr.supervisor.milestone "goal" ──


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from policy_expr.perception import LivePhoneSession
    from policy_expr.schemas import Action, ActionDecision, PolicyTurn

    goal = sys.argv[1] if len(sys.argv) > 1 else "打开微信发一条消息"
    print(f"Goal: {goal}\n")

    sup = MilestoneSupervisorPolicy()
    history: list[PolicyTurn] = []

    with LivePhoneSession() as phone:
        png_bytes = phone.screenshot()
        observation = Observation(png_bytes=png_bytes, source="live")

        log_dir = Path(__file__).parent.parent.parent / "logs" / "policy_expr" / "test"
        log_dir.mkdir(parents=True, exist_ok=True)
        shot_path = log_dir / "screenshot.png"
        shot_path.write_bytes(png_bytes)
        print(f"截图已保存: {shot_path}\n")

        # ── Step 1: 分解 + 首次 evaluate ──
        print("=" * 50)
        print("Step 1: 分解任务 + 首次评估")
        print("=" * 50)
        sv = sup.step(observation, goal, history)
        print(f"should_act     : {sv.should_act}")
        print(f"instruction    : {sv.instruction}")
        print(f"stop           : {sv.stop}")
        print(f"goal_completed : {sv.goal_completed}")
        print(f"summary        : {sv.summary}")
        for mid, m in sup._milestones.items():
            print(f"  [{mid}] {m.name}  status={m.status}  retry={m.retry_count}")

        # 模拟执行一步（用同一个截图，大概率触发 stuck → replan）
        if sv.should_act and sv.instruction:
            history.append(PolicyTurn(
                index=1,
                observation_source="live",
                supervisor=sv,
                action_decision=ActionDecision(
                    action=Action(action_type="tap", x=500, y=500, description="simulated")
                ),
                executed=True,
            ))

            # ── Step 2: 同一张截图 → 大概率 stuck → 触发 replan ──
            print(f"\n{'=' * 50}")
            print("Step 2: 同截图重试（测试 replan）")
            print("=" * 50)
            sv2 = sup.step(observation, goal, history)
            print(f"should_act     : {sv2.should_act}")
            print(f"instruction    : {sv2.instruction}")
            print(f"stop           : {sv2.stop}")
            print(f"goal_completed : {sv2.goal_completed}")
            print(f"summary        : {sv2.summary}")
            current = sup._milestones.get(sup._current_id)
            if current:
                print(f"当前 milestone : [{current.id}] {current.name}  retry={current.retry_count}")

        # ── 打印最终 milestone 状态 ──
        print(f"\n{'=' * 50}")
        print("Milestones 最终状态")
        print("=" * 50)
        for mid, m in sup._milestones.items():
            marker = ">>>" if mid == sup._current_id else "   "
            print(f"{marker} [{mid}] {m.name}  status={m.status}  retry={m.retry_count}")
            print(f"      {m.success_condition}")

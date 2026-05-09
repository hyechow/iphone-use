"""Milestone supervisor v2: single-step and loop milestones as separate state machines."""

import base64
import io
import json
from typing import Literal, Optional

from PIL import Image
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm.structured import invoke_structured
from policy_expr.config import resolve_llm_config
from policy_expr.policies.base import resize_to_logical_png
from policy_expr.schemas import CollectionScope, Milestone, Observation, PolicyTurn, SupervisorStep

load_dotenv()

MAX_RETRIES = 3
STUCK_SCREEN_WINDOW = 3
STUCK_SCREEN_SIMILARITY = 0.95
STUCK_SCREEN_FROZEN = 0.99
STUCK_REPEAT_WINDOW = 3
STUCK_REPEAT_WORD_OVERLAP = 0.85


# ── Schemas ───────────────────────────────────────────────────────────


class _SingleCheckResult(BaseModel):
    """Checker output for single-step milestones (navigation/filter/action/verification/read_once)."""
    status: Literal["done", "in_progress", "stuck"]
    reason: str = Field(description="判断理由")
    stuck_reason: str = Field(default="", description="卡住原因（仅 stuck 时填写）")
    issues: list[str] = Field(default_factory=list)
    visible_evidence: list[str] = Field(default_factory=list, description="截图中支持 done 的可见证据")
    missing_evidence: list[str] = Field(default_factory=list, description="缺失的验收证据")
    summary: str = Field(description="当前屏幕状态一句话描述")
    read_instruction: Optional[str] = Field(
        default=None,
        description="kind=collection(read_once) 或 kind=verification 时填写：当前屏幕需要提取的内容说明；其他类型留空",
    )
    frozen: bool = Field(default=False, description="屏幕是否冻结（相似度≥99%，即使 reader 返回新内容也应停止）")


class _LoopFrameResult(BaseModel):
    """Per-frame assessment for scroll_until_boundary milestones."""
    boundary_reached: bool = Field(default=False, description="当前可见内容是否已到达列表物理边界（无更多条目）")
    should_stop: bool = Field(default=False, description="是否满足停止条件，应结束滚动采集")
    stop_reason: str = Field(default="", description="停止原因（should_stop=true 时填写）")
    read_instruction: str = Field(default="", description="当前屏幕需要提取的内容说明；无相关内容时留空")
    collection_scope: Optional[CollectionScope] = Field(default=None)
    summary: str = Field(description="当前屏幕内容一句话描述")


class _PlanResult(BaseModel):
    instruction: str = Field(description="下一步精确操作指令")
    summary: str = Field(description="规划依据一句话摘要")


class _ReplanResult(BaseModel):
    diagnosis: str = Field(description="失败根本原因（一句话）")
    strategy: Literal["local_replan", "escalate_human"]
    instruction: str = Field(default="")
    escalation_message: str = Field(default="")
    can_degrade_to_collection: bool = Field(default=False)


class _StopConditionPatch(BaseModel):
    scroll_stop_condition: str = Field(
        description="一句话描述何时应停止滚动，例如：「当可见记录日期早于2026-05-03时停止」"
    )


class _DecomposeResponse(BaseModel):
    goal: str
    global_constraints: list[str] = Field(default_factory=list)
    milestones: list[Milestone]
    task_type: Literal["action", "analysis"] = Field(
        description="action=执行具体操作；analysis=查看/比较/总结信息；有疑问时选 analysis"
    )


# ── Prompts ───────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = """\
你是 iPhone 自动化任务的规划 Supervisor。将用户任务分解为子目标（milestone）。
你会收到当前屏幕截图，请根据截图判断设备当前状态。

可用操作：tap（点击）、type（输入）、scroll（滚动）、home（返回主屏幕）

输出要求：
- goal：任务一句话描述
- global_constraints：全局约束列表
- milestones：子目标列表，每个包含 id/name/description/depends_on/success_condition/kind/completion_strategy/scroll_stop_condition/failure_hints
- task_type：
  - action：用户要求执行具体操作（发消息、打开应用、修改设置）
  - analysis：用户要求查看/比较/总结信息（统计数据、总结列表、查询结果）；有疑问时选 analysis

原则：
1. 如果当前不在主屏幕，第一个子目标应为「回到主屏幕」，验收条件为「看到主屏幕（桌面图标界面）」
2. 如果当前已在主屏幕或已在目标应用内，不需要「回到主屏幕」步骤
3. success_condition 要具体可判断，如「看到XX页面标题」「XX输入框有内容」
4. depends_on 填依赖的前置子目标 id，无依赖留空
5. kind 必须表达子目标语义：
   - navigation：打开应用、进入页面、切换 tab
   - filter：设置范围、搜索词、筛选条件、排序条件
   - collection：读取并收集页面内容（记录列表、消息流、搜索结果）
   - action：执行一次具体操作
   - verification：确认结果是否满足目标
6. completion_strategy 必须表达完成方式：
   - visible_once：看到指定页面/状态即可完成
   - read_once：读取当前屏幕一次即可完成
   - scroll_until_boundary：需要反复滚动，直到列表到底或无更多内容
   - repeat_until_satisfied：重复操作直到条件满足
   - human_escalation：需要人工处理
7. 信息获取类任务的内容收集子目标必须使用 kind=collection；来自可滚动列表、记录流或消息流的内容必须使用 completion_strategy=scroll_until_boundary
   - scroll_until_boundary 的子目标必须填写 scroll_stop_condition（一句话说明何时停止滚动）：
     * 有时间范围：「当可见记录日期早于 {目标开始日期} 时停止」
     * 有关键词条件：「当可见内容不再包含 {关键词} 时停止」
     * 全量采集：「滚动至列表物理底部时停止」
8. failure_hints 列出该子目标可能失败的原因
9. 「打开应用」类子目标验收条件应为「成功进入该应用（任意页面均可）」
10. 进入应用内某个 tab/page 的验收条件必须包含可见页面标题、选中状态或该页面独有的稳定内容证据
"""

SINGLE_CHECKER_PROMPT = """\
你是 iPhone 自动化任务的验收员。根据当前屏幕截图和子目标验收条件，判断执行进展。

## 当前子目标
- 名称：{milestone_name}
- 描述：{milestone_desc}
- 验收条件：{success_condition}
- 子目标类型：{milestone_kind}
- 完成策略：{completion_strategy}
- 任务类型：{task_type}
- 全局约束：{constraints}

## 历史操作记录
{history_text}

## 筛选类子目标（kind=filter）
- 截图必须显示精确的筛选条件或等价范围，才能判 done
- 更宽的范围不能当作筛选完成；即使可见项都在目标范围内，筛选摘要显示更宽范围也不能 done
- 如果当前 UI 无法精确筛选，返回 stuck，在 stuck_reason 中说明原因

## 搜索类子目标（kind=filter，含搜索操作）
判 done 必须同时满足：
1. 当前页面是结果页，不是信息流、建议页、历史页或加载页
2. 搜索框或标题显示完整目标查询/条件
3. 页面显示与查询对应的结果列表或详情

## 内容读取（kind=collection read_once 或 kind=verification）
- 如果当前屏幕有与用户目标相关的可提取内容，填写 read_instruction
- navigation/filter/action 阶段 read_instruction 必须留空

## 通用规则
- done：截图上必须能直接观察到验收条件中描述的具体内容
- in_progress：正在向目标推进
- stuck：错误弹窗、连续无进展、页面回退、操作陷入循环
- 验收条件要求某页面标题：必须看到顶部标题与验收条件精确匹配
- 验收条件要求某底部 tab：必须看到该 tab 高亮/选中
- 只看可观测事实，不要凭感觉判断
- done 时：visible_evidence 必须列出截图中直接支持验收条件的文字；missing_evidence 必须为空
- 存在任何 missing_evidence 不能返回 done
"""

LOOP_FRAME_PROMPT = """\
你是内容收集的屏幕状态评估员。当前任务正在滚动收集页面列表内容。
根据当前截图，评估以下内容：

## 当前子目标
- 名称：{milestone_name}
- 描述：{milestone_desc}
- 停止条件：{scroll_stop_condition}
- 全局约束：{constraints}

## 历史操作记录
{history_text}

## 评估要点

### 1. 列表边界（boundary_reached）
boundary_reached=true 必须有明确可见证据，例如：
- 看到"没有更多内容"、"已全部加载"、"到底了"等文字
- 列表末尾出现明显空白且无加载指示器
- 看到与前一屏重叠的最后一条记录，且下方无新内容
不确定时填 false。

### 2. 停止判断（should_stop）
对照上方「停止条件」，判断当前屏幕是否已触发该条件：
- should_stop=true：当前可见内容已满足停止条件，继续滚动只会偏离目标
- should_stop=false：目标内容仍在当前滚动方向，应继续采集
- 如果停止条件是「滚动至列表物理底部」，should_stop 跟随 boundary_reached
- 只有确定触发时才返回 true；不确定时返回 false
- should_stop=true 时必须填写 stop_reason 说明触发依据

### 3. 当前屏幕内容（read_instruction）
如果当前屏幕有与用户目标相关的内容，填写 read_instruction，说明需要提取哪些字段（如时间、金额、名称、状态）。
无相关内容时留空。

### 4. 采集范围（collection_scope，可选）
如果可见内容有明确的范围标志（时间范围、分组标题、筛选摘要），填写 collection_scope 作为参考信息。
"""

PLAN_PROMPT = """\
你是 iPhone 自动化任务的步骤规划器。根据当前截图、子目标和验收结果，给出下一步操作指令。

## 当前子目标
- 名称：{milestone_name}
- 描述：{milestone_desc}
- 验收条件：{success_condition}
- 子目标类型：{milestone_kind}
- 全局约束：{constraints}

## Checker 结果
- status：{check_status}
- reason：{check_reason}
- issues：{issues}
- missing_evidence：{missing_evidence}
- 当前屏幕摘要：{check_summary}

## 历史操作记录
{history_text}

规划规则：
- 只输出一个当前屏幕马上可执行的单步操作指令
- 描述要操作的具体 UI 元素，如「点击底部导航栏左起第二个通讯录图标」
- 不要给出目标级指令，如「进入通讯录页面」「完成搜索」
- 如果当前子目标要求「回到主屏幕」，下一步必须指令「按 Home 键返回主屏幕」
- 如果当前已在目标应用内但不在正确页面，给出应用内导航指令，不要重新打开已打开的应用
- 如果当前在 iOS 主屏幕且目标应用图标不可见：优先点击底部「搜索」胶囊按钮；看不到时才向下滑动打开系统搜索；搜索框出现后输入应用名称
- 如果当前是 iOS 系统搜索页，直接输入或点击搜索结果中的目标应用，不要返回主屏
- 如果当前在应用的子页面需要回到上级，优先使用可见返回控件
- 如果当前在应用内搜索页：搜索框已聚焦则直接输入；未聚焦则先点击搜索框
- 滚动指令描述要查看什么内容（如「滚动查看更早的消息」），不要指定手指方向
"""

LOOP_SCROLL_PROMPT = """\
你是滚动方向决策器。当前任务需要滚动收集页面内容，请根据截图决定第一次滚动的方向和位置。

## 当前子目标
- 名称：{milestone_name}
- 描述：{milestone_desc}
- 全局约束：{constraints}

## 当前屏幕状态
{frame_summary}

规则：
- 输出一个滚动指令，描述要查看什么内容（如「滚动查看更多记录」「滚动查看更早的消息」）
- 不要指定手指滑动方向，由 action policy 根据截图判断
- 如果当前屏幕已显示列表内容，滚动以获取更多同类内容
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

## 分析要求
1. 观察截图，理解当前所有可见 UI 元素
2. 检查历史操作是否存在 A→B→A→B 交替循环，如果存在必须跳出
3. 分析之前失败的根本原因
4. 找到一条不同的路径

## 决策规则
- 工具限制/数据问题 → local_replan
- 如果筛选无法精确设置，但后续 collection 子目标可通过逐条过滤补偿，can_degrade_to_collection=true
- 以下指令已尝试过且失败，禁止再次使用：
{tried_instructions}
- instruction 只包含一个原子操作，禁止包含「并」「然后」「再」等连接词
- 如果子目标要求「回到主屏幕」，必须指令「按 Home 键返回主屏幕」
- 滚动指令描述要查看什么内容，不要指定手指方向
"""

STOP_CONDITION_PATCH_PROMPT = """\
你是 iPhone 自动化任务的规划助手。一个需要滚动采集的子目标需要生成停止条件。

根据用户目标、本子目标的验收条件、以及前置子目标的验收条件，推导出停止滚动的边界。
停止条件必须从验收条件中的筛选维度推导——如果前置验收条件限定了时间范围，停止条件就用时间边界；如果限定了关键词，就用关键词边界。

规则：
- 输出一句话描述何时停止滚动
- 必须从验收条件的约束维度推导，不能泛化为"滚动至列表物理底部"
- 如果验收条件限定时间范围，停止条件必须包含对应日期边界
- 如果验收条件限定关键词/类别，停止条件必须包含对应的消失条件
- 只有没有任何筛选约束的全量采集，才使用"滚动至列表物理底部时停止"
"""

# ── History formatter ─────────────────────────────────────────────────


def _format_history(history: list[PolicyTurn]) -> str:
    if not history:
        return "（无历史记录，这是第一轮）"
    recent = history[-8:]
    lines = []
    for idx, turn in enumerate(recent):
        sv = turn.supervisor
        result = recent[idx + 1].supervisor.summary if idx + 1 < len(recent) else "（结果尚未记录）"
        if turn.action_decision and turn.executed:
            action = turn.action_decision.action
            lines.append(
                f"{turn.index}. 指令=「{sv.instruction}」"
                f" → [{action.action_type}] {action.description}"
                f" → 结果: {result}"
            )
        elif turn.action_decision and not turn.executed:
            action = turn.action_decision.action
            lines.append(
                f"{turn.index}. 指令=「{sv.instruction}」 → [未执行] [{action.action_type}] {action.description}"
            )
        else:
            lines.append(f"{turn.index}. [跳过动作] {sv.summary} → 结果: {result}")
    return "\n".join(lines)


# ── Main class ────────────────────────────────────────────────────────


class MilestoneSupervisorPolicy:
    """Two-machine milestone supervisor: single-step and loop run independently."""

    name = "milestone"

    def __init__(self) -> None:
        self._global_constraints: list[str] = []
        self._milestones: dict[str, Milestone] = {}
        self._order: list[str] = []
        self._current_id: Optional[str] = None
        self._initialized = False
        self._recent_screenshots: list[bytes] = []
        self.task_type: Literal["action", "analysis"] = "action"

    def step(self, observation: Observation, goal: str, history: list[PolicyTurn]) -> SupervisorStep:
        if not self._initialized:
            self._decompose(goal, observation)
            self._initialized = True

        if self._current_id is None:
            return self._terminal_step()

        milestone = self._milestones[self._current_id]
        if _is_loop(milestone):
            return self._run_loop_turn(milestone, observation, history)
        return self._run_single_turn(milestone, observation, history)

    # ── Single-step machine ───────────────────────────────────────────

    def _run_single_turn(
        self,
        milestone: Milestone,
        observation: Observation,
        history: list[PolicyTurn],
    ) -> SupervisorStep:
        sim_stuck = self._check_screen_similarity(observation)
        rep_stuck = self._check_instruction_repetition(history, milestone.id) if not sim_stuck else None
        check = sim_stuck or rep_stuck or self._single_check(milestone, observation, history)
        print(f"  [SingleCheck] {check.status}: {check.reason}")

        if check.status == "done":
            return self._advance(milestone, observation, history)
        if check.status == "stuck":
            return self._handle_stuck(milestone, check, check.read_instruction, observation, history)
        return self._plan_single(milestone, check, observation, history)

    def _plan_single(
        self,
        milestone: Milestone,
        check: _SingleCheckResult,
        observation: Observation,
        history: list[PolicyTurn],
    ) -> SupervisorStep:
        plan = self._invoke_planner(milestone, check, observation, history)
        if self._is_sequence(plan.instruction):
            print("  [Planner] 多步序列，重试...")
            plan = self._invoke_planner(
                milestone, check, observation, history,
                extra="你刚才输出了多个步骤，请只返回当前屏幕上马上要做的一个操作。",
            )
        print(f"  [Planner] {plan.instruction}")
        milestone.status = "running"
        return SupervisorStep(
            should_act=bool(plan.instruction),
            instruction=plan.instruction or None,
            stop=False,
            goal_completed=False,
            summary=plan.summary,
            **_ctx(milestone, check.read_instruction),
        )

    # ── Loop machine ──────────────────────────────────────────────────

    def _run_loop_turn(
        self,
        milestone: Milestone,
        observation: Observation,
        history: list[PolicyTurn],
    ) -> SupervisorStep:
        # Termination: sim_stuck (screen stopped changing)
        sim_stuck = self._check_screen_similarity(observation)
        last_read_added = bool(
            history
            and history[-1].supervisor.milestone_id == milestone.id
            and history[-1].read_added_content
        )
        if sim_stuck:
            if sim_stuck.frozen:
                print("  [Loop] 屏幕冻结（≥99%），即使 reader 返回新内容也结束收集")
                return self._advance(milestone, observation, history)
            if not last_read_added:
                print("  [Loop] 截图连续无变化且无新增内容 → 判为边界，结束收集")
                return self._advance(milestone, observation, history)
            print("  [Loop] 截图相似但上一轮读到了新内容，继续收集")

        # Per-frame assessment
        frame = self._loop_check(milestone, observation, history)
        print(f"  [LoopFrame] boundary={frame.boundary_reached}, should_stop={frame.should_stop}")
        if frame.should_stop:
            print(f"  [Loop] 停止条件触发：{frame.stop_reason}")
        read_inst = frame.read_instruction or _default_read_instruction(milestone)

        # Termination: stop condition triggered
        if frame.should_stop:
            if _has_collected(history, milestone.id):
                print("  [Loop] 已触发停止条件且有采集内容 → 结束收集")
                final_read = _ctx(milestone, read_inst, frame.collection_scope)
                if milestone.scroll_stop_condition:
                    final_read["collection_summary"] = (
                        f"停止条件「{milestone.scroll_stop_condition}」已触发"
                        f"（{frame.stop_reason}）"
                    )
                return self._advance(
                    milestone, observation, history,
                    final_read=final_read,
                )
            stuck = _SingleCheckResult(
                status="stuck",
                reason=f"停止条件已触发但尚未采集到目标内容：{frame.stop_reason}",
                stuck_reason="停止条件触发且没有可用采集结果",
                summary=frame.summary,
            )
            return self._handle_stuck(milestone, stuck, read_inst, observation, history)

        # Termination: boundary confirmed after at least one scroll
        if frame.boundary_reached and _last_scroll_was_for(history, milestone.id):
            print("  [Loop] 确认列表边界 → 结束收集")
            return self._advance(milestone, observation, history)

        # Continue: reuse last scroll or plan first scroll
        milestone.status = "running"
        if _last_scroll_was_for(history, milestone.id):
            return SupervisorStep(
                should_act=True,
                instruction="继续滚动",
                preformed_action=history[-1].action_decision,
                stop=False,
                goal_completed=False,
                summary=f"继续滚动收集内容。{frame.summary}",
                read_instruction=read_inst,
                allow_read=bool(read_inst),
                milestone_id=milestone.id,
                milestone_kind=milestone.kind,
                completion_strategy=milestone.completion_strategy,
                collection_scope=frame.collection_scope,
            )

        plan = self._invoke_loop_scroll(milestone, frame, observation)
        print(f"  [LoopScroll] {plan.instruction}")
        return SupervisorStep(
            should_act=True,
            instruction=plan.instruction,
            stop=False,
            goal_completed=False,
            summary=plan.summary,
            read_instruction=read_inst,
            allow_read=bool(read_inst),
            milestone_id=milestone.id,
            milestone_kind=milestone.kind,
            completion_strategy=milestone.completion_strategy,
            collection_scope=frame.collection_scope,
        )

    # ── Shared: advance, stuck, terminal ─────────────────────────────

    def _advance(
        self,
        milestone: Milestone,
        observation: Observation,
        history: list[PolicyTurn],
        final_read: Optional[dict] = None,
    ) -> SupervisorStep:
        """Mark milestone done, route immediately to next milestone's machine."""
        done_name = milestone.name
        milestone.status = "done"
        self._current_id = self._next_milestone()
        self._recent_screenshots.clear()
        print(f"  子目标「{done_name}」已完成")

        if self._current_id is None:
            return SupervisorStep(
                should_act=False, stop=True, stop_reason="所有子目标已完成",
                goal_completed=True, summary=f"子目标「{done_name}」已完成，任务全部完成。",
                **(final_read or {}),
            )

        next_ms = self._milestones[self._current_id]
        print(f"  开始执行「{next_ms.name}」")

        # If there's content to read from the completed milestone, return an
        # intermediate step so the runner processes it before advancing.
        if final_read:
            return SupervisorStep(
                should_act=False, stop=False, goal_completed=False,
                summary=f"子目标「{done_name}」已完成，下一子目标「{next_ms.name}」待执行。",
                **final_read,
            )

        if _is_loop(next_ms):
            return self._run_loop_turn(next_ms, observation, history)
        return self._run_single_turn(next_ms, observation, history)

    def _handle_stuck(
        self,
        milestone: Milestone,
        check: _SingleCheckResult,
        read_inst: Optional[str],
        observation: Observation,
        history: list[PolicyTurn],
    ) -> SupervisorStep:
        self._recent_screenshots.clear()
        milestone.retry_count += 1

        if milestone.retry_count >= MAX_RETRIES:
            fallback = self._try_filter_fallback(milestone, can_degrade=True, read_inst=read_inst)
            if fallback:
                return fallback
            return self._fail(milestone, check, read_inst)

        print(f"  [Replan] 第 {milestone.retry_count} 次重试...")
        replan = self._invoke_replanner(milestone, check, observation, history)
        print(f"  [Replan] 诊断={replan.diagnosis}, 策略={replan.strategy}")

        if replan.strategy == "escalate_human":
            fallback = self._try_filter_fallback(
                milestone, can_degrade=replan.can_degrade_to_collection, read_inst=read_inst,
            )
            if fallback:
                return fallback
            milestone.status = "failed"
            self._current_id = self._next_milestone()
            return SupervisorStep(
                should_act=False,
                stop=self._current_id is None,
                stop_reason=replan.escalation_message or "升级人工介入",
                goal_completed=False,
                summary=replan.diagnosis,
                **_ctx(milestone, read_inst),
            )

        milestone.status = "running"
        return SupervisorStep(
            should_act=bool(replan.instruction),
            instruction=replan.instruction or None,
            stop=False,
            goal_completed=False,
            summary=f"子目标「{milestone.name}」卡住，第 {milestone.retry_count} 次重试。{replan.diagnosis}",
            **_ctx(milestone, read_inst),
        )

    def _fail(self, milestone: Milestone, check: _SingleCheckResult, read_inst: Optional[str]) -> SupervisorStep:
        milestone.status = "failed"
        self._current_id = self._next_milestone()
        print(f"  子目标「{milestone.name}」失败")
        if self._current_id is None:
            return SupervisorStep(
                should_act=False, stop=True,
                stop_reason=f"子目标「{milestone.name}」重试 {MAX_RETRIES} 次后失败",
                goal_completed=False, summary=check.reason,
                **_ctx(milestone, read_inst),
            )
        return SupervisorStep(
            should_act=False, stop=False, goal_completed=False,
            summary=f"子目标「{milestone.name}」失败，跳过继续下一个。",
            **_ctx(self._milestones[self._current_id], read_inst),
        )

    def _terminal_step(self) -> SupervisorStep:
        failed = [m for m in self._milestones.values() if m.status == "failed"]
        pending = [m for m in self._milestones.values() if m.status == "pending"]
        if failed or pending:
            return SupervisorStep(
                should_act=False, stop=True, goal_completed=False,
                stop_reason=f"无可执行子目标；失败：{'、'.join(m.name for m in failed) or '无'}；未完成：{'、'.join(m.name for m in pending) or '无'}",
                summary="任务未完成，存在失败或依赖未满足的子目标。",
            )
        return SupervisorStep(
            should_act=False, stop=True, stop_reason="所有子目标已完成",
            goal_completed=True, summary="任务完成",
        )

    def _try_filter_fallback(
        self,
        milestone: Milestone,
        can_degrade: bool,
        read_inst: Optional[str],
    ) -> Optional[SupervisorStep]:
        if milestone.kind != "filter" or not can_degrade:
            return None
        dependent = next(
            (self._milestones[mid] for mid in self._order
             if self._milestones[mid].status == "pending"
             and milestone.id in self._milestones[mid].depends_on
             and self._milestones[mid].kind == "collection"),
            None,
        )
        if dependent is None:
            return None
        milestone.status = "done"
        self._current_id = dependent.id
        self._recent_screenshots.clear()
        msg = (
            f"子目标「{milestone.name}」无法精确筛选，已降级为在「{dependent.name}」阶段收集并过滤。"
        )
        if msg not in self._global_constraints:
            self._global_constraints.append(msg)
        print(f"  [Fallback] {msg}")
        return SupervisorStep(
            should_act=False, stop=False, goal_completed=False, summary=msg,
            **_ctx(dependent, read_inst),
        )

    # ── LLM invocations ───────────────────────────────────────────────

    def _single_check(
        self,
        milestone: Milestone,
        observation: Observation,
        history: list[PolicyTurn],
        extra: str = "",
    ) -> _SingleCheckResult:
        prompt = SINGLE_CHECKER_PROMPT.format(
            milestone_name=milestone.name,
            milestone_desc=milestone.description,
            success_condition=milestone.success_condition,
            milestone_kind=milestone.kind,
            completion_strategy=milestone.completion_strategy,
            task_type=self.task_type,
            constraints=json.dumps(self._global_constraints, ensure_ascii=False),
            history_text=_format_history(history),
        )
        if extra:
            prompt += f"\n\n## 输出修正要求\n{extra}"
        result = invoke_structured(self._llm(), self._msgs(prompt, observation), _SingleCheckResult)

        # Guard: done without evidence
        if result.status == "done" and (not result.visible_evidence or result.missing_evidence):
            print("  [SingleCheck] done 缺少证据，重试...")
            result = self._single_check(
                milestone, observation, history,
                extra="你刚才返回 done 但 visible_evidence 为空或 missing_evidence 非空。请重新核对截图，确有证据才能 done，否则返回 in_progress 或 stuck。",
            )
        if result.status == "done" and (not result.visible_evidence or result.missing_evidence):
            return _SingleCheckResult(
                status="stuck",
                reason="checker 返回 done 但缺少可见验收证据",
                stuck_reason="done 缺少可见证据",
                summary=result.summary,
            )
        return result

    def _loop_check(
        self,
        milestone: Milestone,
        observation: Observation,
        history: list[PolicyTurn],
    ) -> _LoopFrameResult:
        prompt = LOOP_FRAME_PROMPT.format(
            milestone_name=milestone.name,
            milestone_desc=milestone.description,
            scroll_stop_condition=milestone.scroll_stop_condition or "滚动至列表物理底部时停止",
            constraints=json.dumps(self._global_constraints, ensure_ascii=False),
            history_text=_format_history(history),
        )
        return invoke_structured(self._llm(), self._msgs(prompt, observation), _LoopFrameResult)

    def _invoke_planner(
        self,
        milestone: Milestone,
        check: _SingleCheckResult,
        observation: Observation,
        history: list[PolicyTurn],
        extra: str = "",
    ) -> _PlanResult:
        prompt = PLAN_PROMPT.format(
            milestone_name=milestone.name,
            milestone_desc=milestone.description,
            success_condition=milestone.success_condition,
            milestone_kind=milestone.kind,
            constraints=json.dumps(self._global_constraints, ensure_ascii=False),
            check_status=check.status,
            check_reason=check.reason,
            issues=json.dumps(check.issues, ensure_ascii=False),
            missing_evidence=json.dumps(check.missing_evidence, ensure_ascii=False),
            check_summary=check.summary,
            history_text=_format_history(history),
        )
        if extra:
            prompt += f"\n\n## 输出修正要求\n{extra}"
        return invoke_structured(self._llm(), self._msgs(prompt, observation), _PlanResult)

    def _invoke_loop_scroll(
        self,
        milestone: Milestone,
        frame: _LoopFrameResult,
        observation: Observation,
    ) -> _PlanResult:
        prompt = LOOP_SCROLL_PROMPT.format(
            milestone_name=milestone.name,
            milestone_desc=milestone.description,
            constraints=json.dumps(self._global_constraints, ensure_ascii=False),
            frame_summary=frame.summary,
        )
        return invoke_structured(self._llm(), self._msgs(prompt, observation), _PlanResult)

    def _invoke_replanner(
        self,
        milestone: Milestone,
        check: _SingleCheckResult,
        observation: Observation,
        history: list[PolicyTurn],
        extra: str = "",
    ) -> _ReplanResult:
        tried = sorted({
            t.supervisor.instruction
            for t in history
            if t.supervisor
            and t.supervisor.instruction
            and t.supervisor.milestone_id == milestone.id
        })
        tried_text = "\n".join(f"  - 「{i}」" for i in tried) if tried else "  （无）"
        prompt = REPLAN_PROMPT.format(
            milestone_name=milestone.name,
            milestone_desc=milestone.description,
            success_condition=milestone.success_condition,
            stuck_reason=check.stuck_reason or check.reason,
            issues=json.dumps(check.issues, ensure_ascii=False),
            retry_count=milestone.retry_count,
            constraints=json.dumps(self._global_constraints, ensure_ascii=False),
            failure_hints=json.dumps(milestone.failure_hints, ensure_ascii=False),
            history_text=_format_history(history),
            tried_instructions=tried_text,
        )
        if extra:
            prompt += f"\n\n## 输出修正要求\n{extra}"
        result = invoke_structured(self._llm(), self._msgs(prompt, observation), _ReplanResult)
        if self._is_sequence(result.instruction):
            print("  [Replan] 多步序列，重试...")
            result = self._invoke_replanner(
                milestone, check, observation, history,
                extra="你刚才输出了多个步骤，请只返回一个原子操作。",
            )
        return result

    # ── Stuck detection ───────────────────────────────────────────────

    def _check_screen_similarity(self, observation: Observation) -> Optional[_SingleCheckResult]:
        self._recent_screenshots.append(observation.png_bytes)
        if len(self._recent_screenshots) > STUCK_SCREEN_WINDOW:
            self._recent_screenshots.pop(0)
        if len(self._recent_screenshots) < STUCK_SCREEN_WINDOW:
            return None

        current = self._recent_screenshots[-1]
        sims = [_png_sim(current, p) for p in self._recent_screenshots[:-1]]
        max_sim = max(sims)
        if all(s >= STUCK_SCREEN_SIMILARITY for s in sims):
            sim_str = ", ".join(f"{s:.2%}" for s in sims)
            frozen = max_sim >= STUCK_SCREEN_FROZEN
            if frozen:
                print(f"  [SimStuck] {sim_str} → 屏幕冻结（≥{STUCK_SCREEN_FROZEN:.0%}）")
            else:
                print(f"  [SimStuck] {sim_str} → 截图连续无变化")
            return _SingleCheckResult(
                status="stuck",
                reason=f"连续 {STUCK_SCREEN_WINDOW} 帧截图相似度 [{sim_str}]，屏幕无实质变化",
                stuck_reason="连续帧高度相似，上一步操作未生效",
                issues=["屏幕像素变化低于阈值"],
                summary="屏幕连续无变化",
                frozen=frozen,
            )
        sim_2back = _png_sim(self._recent_screenshots[-1], self._recent_screenshots[-3])
        sim_adj = _png_sim(self._recent_screenshots[-1], self._recent_screenshots[-2])
        if sim_2back >= STUCK_SCREEN_SIMILARITY and sim_adj < STUCK_SCREEN_SIMILARITY:
            print(f"  [SimStuck] 2back={sim_2back:.2%}, adj={sim_adj:.2%} → AB 循环")
            return _SingleCheckResult(
                status="stuck",
                reason=f"截图在两种状态间交替（2帧前 {sim_2back:.2%}，相邻帧 {sim_adj:.2%}）",
                stuck_reason="屏幕在两种状态间振荡，操作陷入 AB 交替循环",
                issues=["截图在两个视觉状态间交替出现"],
                summary="屏幕在两种状态间振荡",
            )
        return None

    def _check_instruction_repetition(
        self,
        history: list[PolicyTurn],
        milestone_id: str,
    ) -> Optional[_SingleCheckResult]:
        recent_insts = [
            t.supervisor.instruction
            for t in history[-STUCK_REPEAT_WINDOW:]
            if t.supervisor
            and t.supervisor.instruction
            and t.supervisor.milestone_id == milestone_id
        ]
        if len(recent_insts) < STUCK_REPEAT_WINDOW:
            return None
        base_words = set(recent_insts[-1].split())
        sims = [
            len(base_words & set(inst.split())) / max(len(base_words), len(set(inst.split())), 1)
            for inst in recent_insts[:-1]
        ]
        if all(s >= STUCK_REPEAT_WORD_OVERLAP for s in sims):
            sim_str = ", ".join(f"{s:.2%}" for s in sims)
            print(f"  [RepStuck] {sim_str} → 指令连续重复")
            return _SingleCheckResult(
                status="stuck",
                reason=f"连续 {STUCK_REPEAT_WINDOW} 步指令词语重叠 [{sim_str}]，操作策略未变化",
                stuck_reason="连续相似指令，重复操作未生效",
                issues=["supervisor 指令持续重复"],
                summary="操作陷入重复循环",
            )
        return None

    # ── Decompose & routing ───────────────────────────────────────────

    _MAX_DECOMPOSE_RETRIES = 2

    def _decompose(self, goal: str, observation: Observation) -> None:
        cfg = resolve_llm_config("supervisor.decompose")
        if not cfg.model:
            cfg = resolve_llm_config("supervisor")
        print(f"Supervisor: {cfg.provider} / {cfg.model}")
        llm = ChatOpenAI(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)

        # Decompose → validate → retry with feedback if needed
        issues: list[str] = []
        for attempt in range(self._MAX_DECOMPOSE_RETRIES + 1):
            self._do_decompose(llm, goal, observation, issues)
            issues = self._validate_decomposition(goal)
            if not issues:
                break
            if attempt < self._MAX_DECOMPOSE_RETRIES:
                print(f"  [Guard] 分解校验发现 {len(issues)} 项问题，重试 ({attempt+1}/{self._MAX_DECOMPOSE_RETRIES})...")
                for i in issues:
                    print(f"  [Guard]   {i}")

        # Final structural fixes for anything the retry couldn't resolve
        self._patch_decomposition(llm, goal)

        print(f"任务分解为 {len(self._milestones)} 个子目标：")
        for mid in self._order:
            m = self._milestones[mid]
            deps = f" (依赖: {m.depends_on})" if m.depends_on else ""
            machine = "loop" if _is_loop(m) else "single"
            print(f"  [{m.id}][{machine}] {m.name}{deps}")
            print(f"       验收：{m.success_condition}")
            if m.scroll_stop_condition:
                print(f"       停止条件：{m.scroll_stop_condition}")

    def _do_decompose(
        self, llm: ChatOpenAI, goal: str, observation: Observation,
        feedback: list[str],
    ) -> None:
        msgs = self._msgs(DECOMPOSE_PROMPT, observation)
        user_parts: list[dict] = [{"type": "text", "text": f"用户任务：{goal}"}]
        if feedback:
            fb = "\n".join(f"  - {i}" for i in feedback)
            user_parts.append({"type": "text", "text": f"\n上一轮分解存在以下问题，请修正：\n{fb}"})
        msgs[1].content = user_parts + msgs[1].content
        resp = invoke_structured(llm, msgs, _DecomposeResponse)

        self._global_constraints = resp.global_constraints
        self.task_type = resp.task_type
        self._milestones = {m.id: m for m in resp.milestones}
        self._order = [m.id for m in resp.milestones]
        self._current_id = self._next_milestone()

    def _validate_decomposition(self, goal: str) -> list[str]:
        """Check all invariants WITHOUT modifying state. Returns list of issues."""
        issues = []
        all_ids = set(self._milestones.keys())

        # 1. depends_on references must exist
        for m in self._milestones.values():
            for dep in m.depends_on:
                if dep not in all_ids:
                    issues.append(f"子目标「{m.name}」的 depends_on 包含不存在的 ID: {dep}")

        # 2. DAG must not have cycles
        visited: set[str] = set()
        in_stack: set[str] = set()
        def _has_cycle(mid: str) -> bool:
            if mid in in_stack:
                return True
            if mid in visited:
                return False
            visited.add(mid)
            in_stack.add(mid)
            ms = self._milestones.get(mid)
            if ms:
                for dep in ms.depends_on:
                    if _has_cycle(dep):
                        return True
            in_stack.discard(mid)
            return False
        for mid in list(self._order):
            visited.clear()
            in_stack.clear()
            if _has_cycle(mid):
                issues.append(f"子目标之间存在循环依赖（从 {mid} 开始）")

        # 3. success_condition must not be empty
        for m in self._milestones.values():
            if not m.success_condition.strip():
                issues.append(f"子目标「{m.name}」的验收条件为空")

        # 4. kind=collection must pair with read_once or scroll_until_boundary
        for m in self._milestones.values():
            if m.kind == "collection" and m.completion_strategy not in ("read_once", "scroll_until_boundary"):
                issues.append(f"子目标「{m.name}」kind=collection 但 completion_strategy={m.completion_strategy}，应为 read_once 或 scroll_until_boundary")

        # 5. scroll_until_boundary must have scroll_stop_condition
        for m in self._milestones.values():
            if m.completion_strategy == "scroll_until_boundary" and not m.scroll_stop_condition:
                issues.append(f"子目标「{m.name}」使用 scroll_until_boundary 但缺少 scroll_stop_condition")

        # 6. task_type heuristic
        analysis_keywords = ("多少", "什么", "有没有", "查看", "看看", "统计", "查一下", "帮我找", "列出", "汇总", "比较")
        if self.task_type == "action" and any(kw in goal for kw in analysis_keywords):
            issues.append(f"task_type=action 但目标含查询关键词（{', '.join(kw for kw in analysis_keywords if kw in goal)}），应为 analysis")

        return issues

    def _patch_decomposition(self, llm: ChatOpenAI, goal: str) -> None:
        """Apply structural fixes for issues that survive retry. Last resort."""
        fixes = []

        # 1. Remove invalid depends_on
        all_ids = set(self._milestones.keys())
        for m in self._milestones.values():
            invalid = [d for d in m.depends_on if d not in all_ids]
            if invalid:
                m.depends_on = [d for d in m.depends_on if d in all_ids]
                fixes.append(f"子目标「{m.name}」移除无效依赖 {invalid}")

        # 2. Break cycles
        visited: set[str] = set()
        in_stack: set[str] = set()
        def _has_cycle(mid: str) -> bool:
            if mid in in_stack:
                return True
            if mid in visited:
                return False
            visited.add(mid)
            in_stack.add(mid)
            ms = self._milestones.get(mid)
            if ms:
                for dep in ms.depends_on:
                    if _has_cycle(dep):
                        return True
            in_stack.discard(mid)
            return False
        for mid in self._order:
            visited.clear()
            in_stack.clear()
            if _has_cycle(mid):
                self._milestones[mid].depends_on = []
                fixes.append(f"清除子目标「{self._milestones[mid].name}」的依赖以打破循环")

        # 3. Fill empty success_condition
        for m in self._milestones.values():
            if not m.success_condition.strip():
                m.success_condition = f"完成「{m.name}」"
                fixes.append(f"子目标「{m.name}」补全空的验收条件")

        # 4. Fix collection completion_strategy
        for m in self._milestones.values():
            if m.kind == "collection" and m.completion_strategy not in ("read_once", "scroll_until_boundary"):
                m.completion_strategy = "scroll_until_boundary"
                fixes.append(f"子目标「{m.name}」策略修正为 scroll_until_boundary")

        # 5. Fill missing scroll_stop_condition via LLM
        needs_stop_condition = [
            m for m in self._milestones.values()
            if m.completion_strategy == "scroll_until_boundary" and not m.scroll_stop_condition
        ]
        for m in needs_stop_condition:
            dep_context = ""
            if m.depends_on:
                dep_lines = []
                for dep_id in m.depends_on:
                    dep = self._milestones.get(dep_id)
                    if dep:
                        dep_lines.append(f"  - 前置子目标「{dep.name}」验收条件：{dep.success_condition}")
                if dep_lines:
                    dep_context = "\n".join(dep_lines)
            patch = invoke_structured(
                llm,
                [
                    SystemMessage(content=STOP_CONDITION_PATCH_PROMPT),
                    HumanMessage(content=(
                        f"用户目标：{goal}\n"
                        f"子目标名称：{m.name}\n"
                        f"子目标描述：{m.description}\n"
                        f"本子目标验收条件：{m.success_condition}\n"
                        f"{dep_context}\n"
                        f"全局约束：{json.dumps(self._global_constraints, ensure_ascii=False)}"
                    )),
                ],
                _StopConditionPatch,
            )
            m.scroll_stop_condition = patch.scroll_stop_condition
            fixes.append(f"子目标「{m.name}」补全停止条件 → {m.scroll_stop_condition}")

        # 6. Fix task_type
        analysis_keywords = ("多少", "什么", "有没有", "查看", "看看", "统计", "查一下", "帮我找", "列出", "汇总", "比较")
        if self.task_type == "action" and any(kw in goal for kw in analysis_keywords):
            self.task_type = "analysis"
            fixes.append("task_type 从 action 修正为 analysis")

        if fixes:
            print(f"  [Guard] 补丁修复 {len(fixes)} 项：")
            for f in fixes:
                print(f"  [Guard]   {f}")

    def _next_milestone(self) -> Optional[str]:
        for mid in self._order:
            m = self._milestones[mid]
            if m.status != "pending":
                continue
            if all(self._milestones[dep].status == "done" for dep in m.depends_on):
                return mid
        return None

    def _llm(self) -> ChatOpenAI:
        cfg = resolve_llm_config("supervisor")
        return ChatOpenAI(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)

    def _msgs(self, system_prompt: str, observation: Observation) -> list:
        from datetime import datetime
        today = datetime.now().strftime("%Y年%m月%d日 %A")
        b64 = base64.b64encode(resize_to_logical_png(observation.png_bytes)).decode()
        return [
            SystemMessage(content=f"{system_prompt}\n\n当前日期：{today}"),
            HumanMessage(content=[
                {"type": "text", "text": "请根据当前屏幕做出决策。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]),
        ]

    @staticmethod
    def _is_sequence(instruction: str) -> bool:
        text = instruction.strip()
        markers = ("操作序列", "步骤", "\n1.", "\n2.", "1.", "2.", "；2", ";2")
        return any(m in text for m in markers)


# ── Helpers ───────────────────────────────────────────────────────────


def _is_loop(milestone: Milestone) -> bool:
    return (
        milestone.kind == "collection"
        and milestone.completion_strategy == "scroll_until_boundary"
    )


def _last_scroll_was_for(history: list[PolicyTurn], milestone_id: str) -> bool:
    return bool(
        history
        and history[-1].supervisor.milestone_id == milestone_id
        and history[-1].action_decision
        and history[-1].action_decision.action.action_type == "scroll"
        and history[-1].executed
    )


def _has_collected(history: list[PolicyTurn], milestone_id: str) -> bool:
    return any(
        t.supervisor.milestone_id == milestone_id and t.read_added_content
        for t in history
    )


def _default_read_instruction(milestone: Milestone) -> str:
    return (
        f"提取当前屏幕中与「{milestone.name}」相关的所有可见内容，"
        "保留名称/标题、时间/位置、目标相关数值、状态、类别等字段；如果是列表，逐条提取。"
    )


def _ctx(milestone: Milestone, read_instruction: Optional[str], collection_scope=None) -> dict:
    allow_read = milestone.kind in {"collection", "verification"}
    return {
        "read_instruction": read_instruction,
        "allow_read": bool(read_instruction and allow_read),
        "milestone_id": milestone.id,
        "milestone_kind": milestone.kind,
        "completion_strategy": milestone.completion_strategy,
        "collection_scope": collection_scope,
    }


def _png_sim(png1: bytes, png2: bytes, size: int = 64) -> float:
    img1 = Image.open(io.BytesIO(png1)).convert("L").resize((size, size))
    img2 = Image.open(io.BytesIO(png2)).convert("L").resize((size, size))
    total = sum(abs(int(a) - int(b)) for a, b in zip(img1.getdata(), img2.getdata()))
    return 1.0 - total / (255 * size * size)

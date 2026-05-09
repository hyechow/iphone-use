"""Milestone-based supervisor: decompose goal → check → plan/replan."""

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
from policy_expr.schemas import Milestone, Observation, PolicyTurn, SupervisorStep

load_dotenv()

MAX_RETRIES = 3
STUCK_SCREEN_WINDOW = 3           # 连续几帧截图相似才触发
STUCK_SCREEN_SIMILARITY = 0.95    # 像素相似度阈值（0~1）
STUCK_REPEAT_WINDOW = 3           # 连续几步指令相似才触发
STUCK_REPEAT_CHAR_OVERLAP = 0.70  # 指令字符集重叠阈值（0~1）

# ── LLM response schemas ──────────────────────────────────────────────


class _DecomposeResponse(BaseModel):
    goal: str
    global_constraints: list[str] = Field(default_factory=list)
    milestones: list[Milestone]
    task_type: Literal["action", "analysis"] = Field(
        description="action=执行操作类任务；analysis=查找/提取/阅读信息类任务"
    )


class _CheckResult(BaseModel):
    """Checker output: evaluate milestone progress against success_condition."""
    status: str = Field(description="in_progress | done | stuck")
    reason: str = Field(description="判断理由")
    stuck_reason: str = Field(default="", description="卡住原因（仅 stuck 时填写）")
    issues: list[str] = Field(default_factory=list, description="具体问题列表")
    visible_evidence: list[str] = Field(default_factory=list, description="截图中支持 done 的可见证据")
    missing_evidence: list[str] = Field(default_factory=list, description="截图中缺失的验收证据")
    summary: str = Field(description="当前屏幕状态一句话描述")
    read_instruction: Optional[str] = Field(
        default=None,
        description="analysis 任务时填写：当前页面需要提取的具体内容说明，action 任务留空",
    )


class _PlanResult(BaseModel):
    """Planner output: next single-step instruction for an in-progress milestone."""
    instruction: str = Field(description="下一步精确操作指令")
    summary: str = Field(description="规划依据一句话摘要")


class _ReplanResult(BaseModel):
    """Replanner output: diagnose failure and generate fix."""
    diagnosis: str = Field(description="失败根本原因（一句话）")
    strategy: str = Field(description="local_replan | escalate_human")
    instruction: str = Field(default="", description="修复操作指令（local_replan 时必填）")
    escalation_message: str = Field(default="", description="升级人工消息（escalate_human 时填写）")


# ── Prompts ───────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = """\
你是 iPhone 自动化任务的规划 Supervisor。将用户任务分解为子目标（milestone）。
你会收到当前屏幕截图，请根据截图判断设备当前状态。

可用操作：tap（点击）、type（输入）、scroll（滚动）、home（返回主屏幕）

输出要求：
- goal：任务一句话描述
- global_constraints：全局约束列表
- milestones：子目标列表，每个包含 id/name/description/depends_on/success_condition/failure_hints
- task_type：根据用户最终目的判断任务类型（注意：即使是 analysis 任务也需要执行打开应用、滚动等操作来获取信息，判断依据是用户是否需要你「返回信息」）
  - action：用户要你在手机上执行一个具体操作并完成它，如发消息、打开应用、添加联系人、完成设置
  - analysis：用户的目的是获取信息/回答问题/比较内容，如「花了多少钱」「XX和XX哪个便宜」「帮我看看」「总结一下」。即使用户需要你先打开某个应用才能看到内容，仍然是 analysis

原则：
1. 如果当前不在主屏幕，第一个子目标应为「回到主屏幕」，验收条件为「看到主屏幕（桌面图标界面）」
2. 如果当前已在主屏幕或已在目标应用内，不需要「回到主屏幕」步骤
3. 每个子目标 3-8 个操作内可完成
4. success_condition 要具体可判断，如「看到XX页面标题」「XX输入框有内容」
5. depends_on 填依赖的前置子目标 id，无依赖留空
6. 不指定具体 UI 坐标或元素位置（那是 action policy 的事）
7. failure_hints 列出该子目标可能失败的原因
8. "打开应用"类子目标的验收条件应为「成功进入该应用（任意页面均可）」，不要指定必须是主界面——打开 app 可能进入聊天详情、发现页、小程序等任意页面
9. 如果后续操作需要到达应用内特定页面，单独设立导航子目标
10. 进入应用内某个 tab/page 的验收条件必须包含可见页面标题和选中状态；例如微信通讯录必须写成「顶部标题为 通讯录，底部 通讯录 tab 为绿色选中，并显示新的朋友/群聊/标签/公众号等通讯录列表入口」
11. 如果用户目标是信息获取类（看看/查看/了解一下/读一下/帮我总结/都讨论了什么），到达目标页面后应紧跟一个「浏览内容」子目标。验收条件只需要确认已成功进入内容页面即可（如「已进入XX页面，看到内容列表/消息流」），不需要验证是否浏览完所有内容——内容的完整浏览由任务结束时的最终总结生成处理。Planner 在浏览里程碑期间会持续滚动浏览，直到页面到底或截图无变化
"""

CHECKER_PROMPT = """\
你是 iPhone 自动化任务的验收员。根据当前屏幕截图和子目标验收条件，判断执行进展。

## 当前子目标
- 名称：{milestone_name}
- 描述：{milestone_desc}
- 验收条件：{success_condition}
- 任务类型：{task_type}
- 全局约束：{constraints}

## 历史操作记录
{history_text}

## 电商搜索验收规则
当子目标包含「搜索」「查找」等关键词时，适用以下规则：

### 拼多多搜索
- 拼多多首页的特征：底部有多个 tab（首页/推荐等），页面是推荐信息流、轮播图
- 拼多多搜索结果页的特征：顶部搜索框显示完整搜索关键词 + 页面显示商品列表 + 有筛选/排序控件（综合/销量/价格）
- 首页推荐流中出现目标商品 ≠ 搜索结果页，必须判 in_progress
- 搜索框里显示历史搜索词 ≠ 已执行搜索，必须判 in_progress
- 搜索建议页（输入关键词后下拉候选列表）≠ 搜索结果页，必须判 in_progress

### 淘宝搜索
- 淘宝首页的特征：底部 tab 栏、推荐信息流
- 淘宝搜索结果页的特征：顶部搜索框含关键词 + 商品列表 + 筛选/排序栏
- 规则同拼多多

### 通用搜索验收
搜索类子目标判 done 必须同时满足三个条件，缺一不可：
1. 当前页面是搜索结果页（而非首页推荐流、搜索建议页、启动广告等）
2. 搜索框中显示完整搜索关键词
3. 页面显示搜索结果列表和常见控件（筛选、排序等）
仅满足部分条件时应返回 in_progress，并在 missing_evidence 中说明缺少哪些。

### Forbidden States
当子目标包含「搜索」时，以下情况即使出现目标关键词也禁止判 done：
- 各平台首页推荐流 —— 推荐流不是搜索结果
- 搜索建议页 / 搜索历史页 —— 只是输入过程中的中间状态
- 启动广告 / 弹窗 / 引导页 —— 还没到搜索页面

## 浏览类子目标
当子目标名称包含「浏览」「阅读」「查看内容」等关键词时：
- 即使验收条件描述的页面已进入，如果页面明显还有更多内容可以滚动查看（如只显示了部分列表、聊天记录上方还有更早消息），应返回 in_progress 而非 done
- 只有当页面到达滚动边界（底部/顶部无更多内容），或截图显示列表末尾提示（如「没有更多了」），才返回 done
- 浏览期间如果误入其他页面（如点进了某个子页面），应返回 in_progress 而非 done

## 通用判断规则
- done：屏幕上必须能直接观察到验收条件中描述的具体内容。如果无法在截图中确认验收条件已满足，应判为 in_progress 而非 done
- stuck：出现错误弹窗、连续无进展、页面回退、与上一次截图无明显变化；或历史记录显示操作陷入循环
- in_progress：正在向目标推进
- 如果验收条件要求某个页面标题，必须看到顶部标题与验收条件精确匹配
- 如果验收条件要求某个底部 tab，必须看到该 tab 被高亮/选中
- 不要凭感觉判断，只看可观测的事实
- 不要仅凭历史记录判断完成，必须截图上有对应的视觉证据
- issues 列出具体观察到的问题
- status 为 done 时，visible_evidence 必须列出截图中直接支持验收条件的文字或状态；missing_evidence 必须为空
- 如果存在任何 missing_evidence，不能返回 done，应返回 in_progress 或 stuck
- 区分「应用未成功启动」和「应用已打开但不在预期页面」两种情况

输出 JSON：
- status: in_progress | done | stuck
- reason: 判断理由
- stuck_reason: 卡住原因（仅 stuck 时填写）
- issues: 具体问题列表
- visible_evidence: 截图中支持 done 的可见证据列表（done 时必填）
- missing_evidence: 截图中缺失的验收证据列表
- summary: 当前屏幕状态一句话描述
- read_instruction: 仅 task_type=analysis 时填写——描述当前页面需要提取的具体内容（如"提取页面上所有商品名称和价格"）；task_type=action 时必须留空
"""

PLAN_PROMPT = """\
你是 iPhone 自动化任务的步骤规划器。根据当前截图、子目标和 checker 的验收结果，给出下一步操作指令。

## 当前子目标
- 名称：{milestone_name}
- 描述：{milestone_desc}
- 验收条件：{success_condition}
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
- 只输出一个当前屏幕马上可执行的单步操作指令，不能输出编号列表、操作序列或多个步骤
- 如果截图显示验收条件已完全满足、无需任何操作，instruction 返回「停止操作」，action policy 会将其映射为 stop action
- 描述要操作的具体 UI 元素，如「点击底部导航栏左起第二个通讯录图标」
- 不要给出目标级指令，如「进入通讯录页面」「完成搜索」
- 如果当前子目标名称或验收条件明确要求「回到主屏幕 / 返回主屏幕 / 看到主屏幕」，下一步必须指令「按 Home 键返回主屏幕」，不要点击应用内左上角返回按钮
- 如果当前已在目标应用内但不在正确页面，给出应用内导航指令，不要重复尝试打开已打开的应用
- 如果当前在 iOS 主屏幕且目标应用图标不可见，不要猜测图标位置，也不要优先左右翻页；应使用系统搜索找应用：如果底部可见「搜索」胶囊按钮，先指令「点击主屏幕底部的搜索按钮」；只有看不到该按钮时才指令「在主屏幕中部向下滑动打开系统搜索」。看到系统搜索框后再指令输入目标应用名称，看到搜索结果后再点击目标应用
- 如果当前是 iOS 系统搜索页（顶部搜索框、下方显示 Siri 建议/搜索结果/App 图标），打开应用类任务应优先输入或点击搜索结果中的目标应用，不要返回主屏重新找图标
- 如果当前在应用的子页面（如微信聊天详情页，底部显示输入框工具栏而非导航 tab），下一步应指令「点击左上角返回按钮」回到上一级列表页，而非直接要求点击导航 tab（子页面不显示导航 tab）
- 微信搜索页可能显示「搜索本地或网络结果」「AI搜索」「最近在搜」「页面设置」，这仍然是微信内搜索页，不要要求退出重开微信
- 如果当前在微信搜索页且搜索框已有光标，搜索类任务应优先指令「输入搜索关键词」
- 如果当前在微信搜索页但搜索框未聚焦，搜索类任务应指令「点击顶部搜索框」
- 输入框显示灰色占位文字（placeholder）时，说明输入框为空，直接点击后输入即可，不需要先删除占位文字
- 滚动指令不要指定手指滑动方向（不要说「向上滑动」「向下滑动」），而是描述要查看什么内容（如「滚动查看更早的消息」「滚动查看更多商品」），让 action policy 根据截图判断正确方向

输出 JSON：
- instruction: 下一步精确操作指令
- summary: 规划依据一句话摘要
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
- 如果失败子目标名称或验收条件明确要求「回到主屏幕 / 返回主屏幕 / 看到主屏幕」，local_replan 必须指令「按 Home 键返回主屏幕」；不要继续点击应用内返回按钮，也不要点击搜索、确认、商品等应用内控件
- 区分「应用未成功启动」和「应用已打开但停在错误页面」，后者应通过应用内导航解决而非重新打开应用
- instruction 必须只包含一个可执行操作，不能输出编号列表、操作序列或多个步骤
- 如果当前在微信搜索页且搜索框已有光标，优先给出单步修复指令「输入搜索关键词」
- 微信搜索页可能显示「搜索本地或网络结果」「AI搜索」「最近在搜」「页面设置」，这不是 iOS 系统搜索，不要要求退出重开微信
- 输入框显示灰色占位文字（placeholder）时，说明输入框为空，直接点击后输入即可，不需要先删除占位文字
- 滚动指令不要指定手指滑动方向（不要说「向上滑动」「向下滑动」），而是描述要查看什么内容（如「滚动查看更早的消息」「滚动查看更多商品」），让 action policy 根据截图判断正确方向

输出 JSON：
- diagnosis: 失败根本原因（一句话）
- strategy: local_replan | escalate_human
- instruction: 修复操作指令（local_replan 时必填）
- escalation_message: 升级人工消息（escalate_human 时填写）
"""


# ── History formatter (reuse pattern from simple.py) ──────────────────


def _format_history(history: list[PolicyTurn]) -> str:
    """格式：指令 → 执行动作 → 动作结果（下一轮屏幕状态）。

    每条记录包含三个关键信息：
    - 指令：supervisor 要求 action policy 做什么
    - 操作：action policy 实际执行了什么（可能与指令不一致）
    - 结果：执行后屏幕变成什么状态（取自下一轮 supervisor.summary）
    """
    if not history:
        return "（无历史记录，这是第一轮）"

    recent = history[-8:]
    lines = []
    for idx, turn in enumerate(recent):
        sv = turn.supervisor
        # 结果 = 下一轮的屏幕摘要（即本轮动作执行后的屏幕状态）
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


# ── Implementation ────────────────────────────────────────────────────


class MilestoneSupervisorPolicy:
    """Supervisor that decomposes goals into milestones and tracks progress.

    Flow:
    1. step(): decompose if needed → checker evaluates current milestone
    2. done → next milestone; in_progress → planner; stuck → replanner
    3. Checker only validates progress; planner/replanner generate actions
    """

    name = "milestone"

    def __init__(self) -> None:
        self._global_constraints: list[str] = []
        self._milestones: dict[str, Milestone] = {}
        self._order: list[str] = []
        self._current_id: Optional[str] = None
        self._initialized = False
        self._recent_screenshots: list[bytes] = []  # 用于截图相似度检测
        self.task_type: Literal["action", "analysis"] = "action"

    def step(
        self,
        observation: Observation,
        goal: str,
        history: list[PolicyTurn],
    ) -> SupervisorStep:
        if not self._initialized:
            self._decompose(goal, observation)
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

        # ── Checker: evaluate milestone progress only ──
        # 硬逻辑兜底：截图无变化 / 指令循环 → 直接判 stuck，跳过 LLM
        sim_stuck = self._check_screen_similarity(observation)
        is_browsing = any(kw in milestone.name for kw in ["浏览", "阅读", "查看", "阅读内容", "浏览内容"])
        if sim_stuck and is_browsing:
            # 浏览里程碑：页面连续无变化 = 已到内容边界 = 浏览完成
            sim_stuck = _CheckResult(
                status="done",
                reason="页面连续无变化，已浏览到内容边界",
                visible_evidence=["页面连续多帧截图无变化，已到滚动边界"],
                missing_evidence=[],
                summary="页面内容已到底",
            )
        rep_stuck = self._check_instruction_repetition(history) if not sim_stuck and not is_browsing else None
        check = sim_stuck or rep_stuck or self._check(milestone, observation, history)
        print(f"  [Checker] {check.status}: {check.reason}")

        if check.status == "done":
            done_name = milestone.name
            milestone.status = "done"
            self._current_id = self._next_milestone()
            self._recent_screenshots.clear()
            print(f"  子目标「{done_name}」已完成")

            if self._current_id is None:
                return SupervisorStep(
                    should_act=False,
                    stop=True,
                    stop_reason="所有子目标已完成",
                    goal_completed=True,
                    summary=f"子目标「{done_name}」已完成，任务全部完成。",
                )

            # 直接为新 milestone 规划第一步，下一轮再让 checker 验收
            next_ms = self._milestones[self._current_id]
            synthetic_check = _CheckResult(
                status="in_progress",
                reason=f"子目标「{done_name}」已完成，开始执行「{next_ms.name}」",
                summary=check.summary,
            )
            plan = self._plan(next_ms, synthetic_check, observation, history)
            if self._instruction_looks_like_sequence(plan.instruction):
                print("  [Planner] instruction 是多步序列，重试生成单步指令...")
                plan = self._plan(
                    next_ms, synthetic_check, observation, history,
                    extra_instruction=(
                        "你刚才输出了多个步骤，但执行器一次只能执行一个 action。"
                        "请只返回当前屏幕上马上要做的一个操作，不要编号，不要包含后续步骤。"
                    ),
                )
            if self._instruction_is_stop(plan.instruction):
                print("  [Checker] planner 判定无需操作，立即验收当前子目标...")
                next_check = self._check(next_ms, observation, history)
                print(f"  [Checker] {next_check.status}: {next_check.reason}")
                if next_check.status == "done":
                    next_name = next_ms.name
                    next_ms.status = "done"
                    self._current_id = self._next_milestone()
                    self._recent_screenshots.clear()
                    print(f"  子目标「{next_name}」已完成")
                    if self._current_id is None:
                        return SupervisorStep(
                            should_act=False,
                            stop=True,
                            stop_reason="所有子目标已完成",
                            goal_completed=True,
                            summary=(
                                f"子目标「{done_name}」已完成，"
                                f"子目标「{next_name}」也已满足，任务全部完成。"
                            ),
                        )
                    following = self._milestones[self._current_id]
                    return SupervisorStep(
                        should_act=False,
                        stop=False,
                        goal_completed=False,
                        summary=(
                            f"子目标「{done_name}」已完成，"
                            f"子目标「{next_name}」也已满足，开始执行「{following.name}」。"
                        ),
                    )
                plan = self._plan(
                    next_ms,
                    next_check,
                    observation,
                    history,
                    extra_instruction=(
                        "你刚才判断无需操作，但 checker 未验收通过。"
                        "请基于 missing_evidence 和当前截图，返回一个真实可执行的单步操作，不要返回停止操作。"
                    ),
                )
            next_ms.status = "running"
            return SupervisorStep(
                should_act=bool(plan.instruction),
                instruction=plan.instruction or None,
                stop=False,
                goal_completed=False,
                summary=f"子目标「{done_name}」已完成。{plan.summary}",
            )

        if check.status == "stuck":
            self._recent_screenshots.clear()  # replan 后给新策略干净的窗口
            milestone.retry_count += 1

            if milestone.retry_count >= MAX_RETRIES:
                milestone.status = "failed"
                self._current_id = self._next_milestone()
                print(f"  子目标「{milestone.name}」失败")

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

            # ── Replanner ──
            print(f"  [Replanner] 第 {milestone.retry_count} 次重试...")
            replan = self._replan(milestone, check, observation, history)
            if self._instruction_looks_like_sequence(replan.instruction):
                print("  [Replanner] instruction 是多步序列，重试生成单步指令...")
                replan = self._replan(
                    milestone,
                    check,
                    observation,
                    history,
                    extra_instruction=(
                        "你刚才输出了多个步骤，但执行器一次只能执行一个 action。"
                        "请只返回当前屏幕上马上要做的一个操作，不要编号，不要包含后续步骤。"
                    ),
                )
            print(f"  [Replanner] 诊断: {replan.diagnosis}, 策略: {replan.strategy}")

            if replan.strategy == "escalate_human":
                milestone.status = "failed"
                self._current_id = self._next_milestone()
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

        # in_progress ── Planner generates the next single-step instruction ──
        # 浏览里程碑：如果上一轮是滚动动作，直接重复，跳过 Planner 和 Action Policy
        last_was_scroll = (
            history
            and history[-1].action_decision
            and history[-1].action_decision.action.action_type == "scroll"
            and history[-1].executed
        )
        if is_browsing and last_was_scroll:
            print("  [Browsing] 重复上次滚动，跳过 Planner 和 Action Policy")
            milestone.status = "running"
            return SupervisorStep(
                should_act=True,
                instruction="继续滚动",
                stop=False,
                goal_completed=False,
                summary="浏览里程碑：继续滚动浏览内容",
                preformed_action=history[-1].action_decision,
            )

        plan = self._plan(milestone, check, observation, history)
        if self._instruction_looks_like_sequence(plan.instruction):
            print("  [Planner] instruction 是多步序列，重试生成单步指令...")
            plan = self._plan(
                milestone,
                check,
                observation,
                history,
                extra_instruction=(
                    "你刚才输出了多个步骤，但执行器一次只能执行一个 action。"
                    "请只返回当前屏幕上马上要做的一个操作，不要编号，不要包含后续步骤。"
                ),
            )
        print(f"  [Planner] instruction: {plan.instruction}")
        milestone.status = "running"
        return SupervisorStep(
            should_act=bool(plan.instruction),
            instruction=plan.instruction or None,
            stop=False,
            goal_completed=False,
            summary=plan.summary,
            read_instruction=check.read_instruction or None,
        )

    # ── Internal helpers ──────────────────────────────────────────────

    def _decompose(self, goal: str, observation: Observation) -> None:
        cfg = resolve_llm_config("supervisor")
        print(f"Supervisor Provider : {cfg.provider}")
        print(f"Supervisor Model    : {cfg.model}")

        llm = ChatOpenAI(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)
        messages = self._build_messages(DECOMPOSE_PROMPT, observation)
        messages[1].content.insert(
            0,
            {"type": "text", "text": f"用户任务：{goal}"},
        )
        resp = invoke_structured(llm, messages, _DecomposeResponse)

        self._global_constraints = resp.global_constraints
        self.task_type = resp.task_type
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

    def _check_screen_similarity(self, observation: Observation) -> Optional[_CheckResult]:
        """若最近 STUCK_SCREEN_WINDOW 帧截图高度相似，直接返回 stuck，否则返回 None。"""
        self._recent_screenshots.append(observation.png_bytes)
        if len(self._recent_screenshots) > STUCK_SCREEN_WINDOW:
            self._recent_screenshots.pop(0)

        if len(self._recent_screenshots) < STUCK_SCREEN_WINDOW:
            return None

        current = self._recent_screenshots[-1]
        sims = [
            _png_similarity(current, prev)
            for prev in self._recent_screenshots[:-1]
        ]
        if all(s >= STUCK_SCREEN_SIMILARITY for s in sims):
            sim_str = ", ".join(f"{s:.2%}" for s in sims)
            print(f"  [Similarity] {sim_str} → 截图连续无变化，判为 stuck")
            return _CheckResult(
                status="stuck",
                reason=f"连续 {STUCK_SCREEN_WINDOW} 帧截图相似度 [{sim_str}]，屏幕无实质变化",
                stuck_reason="连续帧高度相似，上一步操作未生效",
                issues=["屏幕像素变化低于阈值"],
                visible_evidence=[],
                missing_evidence=[],
                summary="屏幕连续无变化",
            )

        # Period-2 oscillation: current frame similar to 2-steps-back but different from adjacent
        sim_2back = _png_similarity(self._recent_screenshots[-1], self._recent_screenshots[-3])
        sim_adj   = _png_similarity(self._recent_screenshots[-1], self._recent_screenshots[-2])
        if sim_2back >= STUCK_SCREEN_SIMILARITY and sim_adj < STUCK_SCREEN_SIMILARITY:
            print(
                f"  [Similarity] 2-back={sim_2back:.2%}, adj={sim_adj:.2%}"
                f" → 截图 AB 交替循环，判为 stuck"
            )
            return _CheckResult(
                status="stuck",
                reason=f"截图在两种状态间交替（2帧前相似度 {sim_2back:.2%}，相邻帧 {sim_adj:.2%}）",
                stuck_reason="屏幕在两种状态间振荡，操作陷入 AB 交替循环",
                issues=["截图在两个视觉状态间交替出现"],
                visible_evidence=[],
                missing_evidence=[],
                summary="屏幕在两种状态间振荡",
            )
        return None

    def _check_instruction_repetition(
        self, history: list[PolicyTurn],
    ) -> Optional[_CheckResult]:
        """若最近 STUCK_REPEAT_WINDOW 步的 supervisor 指令字符集高度重叠，直接返回 stuck。"""
        if len(history) < STUCK_REPEAT_WINDOW:
            return None

        recent = history[-STUCK_REPEAT_WINDOW:]
        instructions = [
            t.supervisor.instruction
            for t in recent
            if t.supervisor and t.supervisor.instruction
        ]
        if len(instructions) < STUCK_REPEAT_WINDOW:
            return None

        base = set(instructions[-1])
        sims = [
            len(base & set(inst)) / max(len(base), len(set(inst)), 1)
            for inst in instructions[:-1]
        ]
        if all(s >= STUCK_REPEAT_CHAR_OVERLAP for s in sims):
            sim_str = ", ".join(f"{s:.2%}" for s in sims)
            print(f"  [Repetition] {sim_str} → 指令连续重复，判为 stuck")
            return _CheckResult(
                status="stuck",
                reason=f"连续 {STUCK_REPEAT_WINDOW} 步指令字符重叠 [{sim_str}]，操作策略未变化",
                stuck_reason="连续相似指令，重复操作未生效",
                issues=["supervisor 指令持续重复"],
                visible_evidence=[],
                missing_evidence=[],
                summary="操作陷入重复循环",
            )
        return None

    def _check(
        self, milestone: Milestone, observation: Observation, history: list[PolicyTurn],
    ) -> _CheckResult:
        """Checker: evaluate milestone progress against success_condition."""
        check = self._invoke_checker(milestone, observation, history)
        if self._invalid_done_evidence(check):
            print("  [Checker] done 缺少可见证据或存在缺失证据，重试一次...")
            check = self._invoke_checker(
                milestone,
                observation,
                history,
                extra_instruction=(
                    "你刚才返回了 status=done，但 visible_evidence 为空或 missing_evidence 非空。"
                    "这是无效输出。请严格核对截图：只有验收条件中的页面标题、选中 tab、关键列表入口都直接可见时才能 done；"
                    "如果任一证据缺失，返回 in_progress 或 stuck。"
                ),
            )

        if self._invalid_done_evidence(check):
            return _CheckResult(
                status="stuck",
                reason="checker 返回 done 但缺少可见验收证据",
                stuck_reason="done 缺少可见证据",
                issues=[
                    "done 状态必须包含 visible_evidence",
                    "done 状态不能包含 missing_evidence",
                ],
                visible_evidence=check.visible_evidence,
                missing_evidence=check.missing_evidence or ["缺少可见验收证据"],
                summary=check.summary,
            )
        return check

    def _invoke_checker(
        self,
        milestone: Milestone,
        observation: Observation,
        history: list[PolicyTurn],
        extra_instruction: str = "",
    ) -> _CheckResult:
        llm = self._make_checker_llm()
        prompt = CHECKER_PROMPT.format(
            milestone_name=milestone.name,
            milestone_desc=milestone.description,
            success_condition=milestone.success_condition,
            task_type=self.task_type,
            constraints=json.dumps(self._global_constraints, ensure_ascii=False),
            history_text=_format_history(history),
        )
        if extra_instruction:
            prompt = f"{prompt}\n\n## 输出修正要求\n{extra_instruction}"
        messages = self._build_messages(
            prompt,
            observation,
        )
        return invoke_structured(llm, messages, _CheckResult)

    @staticmethod
    def _invalid_done_evidence(check: _CheckResult) -> bool:
        return check.status == "done" and (
            not check.visible_evidence or bool(check.missing_evidence)
        )

    @staticmethod
    def _instruction_is_stop(instruction: str) -> bool:
        text = instruction.strip()
        return text in {"停止操作", "无需操作", "目标已完成"} or "无需任何操作" in text

    def _plan(
        self,
        milestone: Milestone,
        check: _CheckResult,
        observation: Observation,
        history: list[PolicyTurn],
        extra_instruction: str = "",
    ) -> _PlanResult:
        """Planner: generate one executable instruction for an in-progress milestone."""
        llm = self._make_planner_llm()
        prompt = PLAN_PROMPT.format(
            milestone_name=milestone.name,
            milestone_desc=milestone.description,
            success_condition=milestone.success_condition,
            constraints=json.dumps(self._global_constraints, ensure_ascii=False),
            check_status=check.status,
            check_reason=check.reason,
            issues=json.dumps(check.issues, ensure_ascii=False),
            missing_evidence=json.dumps(check.missing_evidence, ensure_ascii=False),
            check_summary=check.summary,
            history_text=_format_history(history),
        )
        if extra_instruction:
            prompt = f"{prompt}\n\n## 输出修正要求\n{extra_instruction}"
        messages = self._build_messages(prompt, observation)
        return invoke_structured(llm, messages, _PlanResult)

    def _replan(
        self, milestone: Milestone, check: _CheckResult,
        observation: Observation, history: list[PolicyTurn],
        extra_instruction: str = "",
    ) -> _ReplanResult:
        """Replanner: diagnose failure and generate fix strategy."""
        llm = self._make_replanner_llm()
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
        )
        if extra_instruction:
            prompt = f"{prompt}\n\n## 输出修正要求\n{extra_instruction}"
        messages = self._build_messages(prompt, observation)
        return invoke_structured(llm, messages, _ReplanResult)

    @staticmethod
    def _instruction_looks_like_sequence(instruction: str) -> bool:
        text = instruction.strip()
        if not text:
            return False
        sequence_markers = ("操作序列", "步骤", "\n1.", "\n2.", "1.", "2.", "；2", ";2")
        return any(marker in text for marker in sequence_markers)

    def _make_checker_llm(self) -> ChatOpenAI:
        return self._make_supervisor_llm()

    def _make_planner_llm(self) -> ChatOpenAI:
        return self._make_supervisor_llm()

    def _make_replanner_llm(self) -> ChatOpenAI:
        return self._make_supervisor_llm()

    @staticmethod
    def _make_supervisor_llm() -> ChatOpenAI:
        cfg = resolve_llm_config("supervisor")
        return ChatOpenAI(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)

    def _build_messages(
        self, system_prompt: str, observation: Observation,
    ) -> list:
        """Build multimodal messages with system prompt + screenshot."""
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


def _png_similarity(png1: bytes, png2: bytes, size: int = 64) -> float:
    """Return pixel-level similarity between two PNGs (0=different, 1=identical).

    Both images are converted to grayscale and resized to `size`×`size` before
    comparison, so the result is resolution-independent and fast.
    """
    img1 = Image.open(io.BytesIO(png1)).convert("L").resize((size, size))
    img2 = Image.open(io.BytesIO(png2)).convert("L").resize((size, size))
    pixels1 = img1.getdata()
    pixels2 = img2.getdata()
    total_diff = sum(abs(int(a) - int(b)) for a, b in zip(pixels1, pixels2))
    return 1.0 - total_diff / (255 * size * size)

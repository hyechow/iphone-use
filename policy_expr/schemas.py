"""Shared schemas for policy experiments."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    """A single phone action in normalized coordinates."""

    action_type: str = Field(
        description="操作类型：tap（纯点击）、type（点击输入框并输入文字）、scroll（滚动）、home（返回主屏幕）之一"
    )
    x: Optional[float] = Field(
        default=None,
        description="归一化 x 坐标（0-1000，tap/type 时必填）",
    )
    y: Optional[float] = Field(
        default=None,
        description="归一化 y 坐标（0-1000，tap/type 时必填）",
    )
    direction: Optional[str] = Field(
        default=None,
        description="滚动方向：up（向上滚动，查看更多内容）或 down（向下滚动，回到顶部），scroll 时必填",
    )
    text: Optional[str] = Field(
        default=None,
        description="要输入的文字内容（action_type 为 type 时必填）",
    )
    description: str = Field(description="该操作的中文说明，如「点击微信图标」")


class Observation(BaseModel):
    """Raw environment observation used by policies."""

    png_bytes: bytes = Field(description="当前 iPhone 截图 PNG bytes")
    source: str = Field(description="观测来源")


class ActionDecision(BaseModel):
    """Action policy output: screen interpretation plus the next action."""

    screen_type: str = Field(
        description="屏幕类型，如 home_screen、app_page、settings、dialog 等"
    )
    app_name: Optional[str] = Field(
        default=None,
        description="当前前台应用名称，主屏幕时为 null",
    )
    summary: str = Field(description="屏幕内容的简要中文描述（1-2句话）")
    reasoning: str = Field(
        description="简短推理：当前界面状态 → 下一步最优先的操作是什么 → 目标元素的坐标判断依据"
    )
    action: Action = Field(description="经过推理后，当前最应该执行的一个操作")


class SupervisorStep(BaseModel):
    """Supervisor policy decision for one turn."""

    should_act: bool = Field(description="是否调用 action policy 执行动作")
    instruction: Optional[str] = Field(
        default=None,
        description="给 action policy 的精确操作指令（should_act=true 时必填）",
    )
    stop: bool = Field(description="是否终止 agent loop")
    stop_reason: str = Field(default="", description="终止原因（stop=true 时填写）")
    goal_completed: bool = Field(description="用户目标是否已完全达成")
    summary: str = Field(description="对当前屏幕状态和任务进展的简要描述")


class PolicyTurn(BaseModel):
    """One observe-decide-act turn saved in continue mode."""

    index: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    observation_source: str
    supervisor: SupervisorStep
    action_decision: Optional[ActionDecision] = None
    executed: bool = False


class PolicyContext(BaseModel):
    """Persistent context for multi-turn policy experiments."""

    goal: str
    supervisor_policy_name: str
    action_policy_name: str
    turns: list[PolicyTurn] = Field(default_factory=list)
    output: Optional[str] = None

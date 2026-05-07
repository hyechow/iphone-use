"""Shared schemas for policy experiments."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal["tap", "type", "scroll", "home"]


class Action(BaseModel):
    """A single phone action in normalized coordinates."""

    action_type: ActionType = Field(
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
        description="滚动方向：up（向上滚动，查看更多内容）、down（向下滚动，回到顶部）、left（向左滑动，如主屏翻到下一页）、right（向右滑动，如主屏翻到上一页），scroll 时必填",
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
    """Action policy output: the next action to execute."""

    action: Action = Field(description="当前应该执行的操作")


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
    app_name: Optional[str] = Field(default=None, description="当前前台应用名称")
    summary: str = Field(description="对当前屏幕状态和任务进展的简要描述")


class Milestone(BaseModel):
    """A sub-goal in the task decomposition DAG."""

    id: str
    name: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    success_condition: str
    failure_hints: list[str] = Field(default_factory=list)
    status: str = Field(default="pending", description="pending | running | done | failed")
    retry_count: int = 0


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

"""Shared schemas for policy experiments."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


ActionType = Literal["tap", "click", "type", "scroll", "home", "stop"]
TaskType = Literal["action", "analysis"]


class Action(BaseModel):
    """A single phone action in normalized coordinates."""

    @model_validator(mode="before")
    @classmethod
    def _unpack_coords(cls, data: object) -> object:
        """Unpack x: [x, y] into separate x/y fields (model sometimes uses list coords)."""
        if isinstance(data, dict):
            x = data.get("x")
            if isinstance(x, list) and len(x) == 2 and "y" not in data:
                return {**data, "x": x[0], "y": x[1]}
            # 常见 LLM 别名：click → tap
            if data.get("action_type") == "click":
                data["action_type"] = "tap"
            # Clamp scroll y to avoid edge dead zones
            if data.get("action_type") == "scroll" and "y" in data and data["y"] is not None:
                data["y"] = max(200, min(float(data["y"]), 850))
        return data

    action_type: ActionType = Field(
        description="操作类型：tap（纯点击）、type（点击输入框并输入文字）、scroll（滚动）、home（返回主屏幕）之一"
    )
    x: Optional[float] = Field(
        default=None,
        description="归一化 x 坐标（0-1000，tap/type/scroll 时必填）",
    )
    y: Optional[float] = Field(
        default=None,
        description="归一化 y 坐标（0-1000，tap/type/scroll 时必填）",
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

    @model_validator(mode="before")
    @classmethod
    def _unwrap_flat_action(cls, data: object) -> object:
        """Accept flat Action fields directly (model forgot the outer wrapper)."""
        if isinstance(data, dict) and "action_type" in data and "action" not in data:
            return {"action": data}
        return data


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
    preformed_action: Optional[ActionDecision] = Field(
        default=None,
        description="预生成的动作决策（设置后 runner 跳过 Action Policy 直接执行）",
    )
    read_instruction: Optional[str] = Field(
        default=None,
        description="当前屏幕需要提取的内容说明（analysis 任务时由 Checker 填写）",
    )


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
    llm_calls: int = 0


class PolicyContext(BaseModel):
    """Persistent context for multi-turn policy experiments."""

    goal: str
    supervisor_policy_name: str
    action_policy_name: str
    turns: list[PolicyTurn] = Field(default_factory=list)
    task_type: Optional[TaskType] = None
    content_notes: list[str] = Field(default_factory=list)
    output: Optional[str] = None

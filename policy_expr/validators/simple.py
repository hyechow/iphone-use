"""Simple LLM-based turn validator."""

import base64

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from llm.structured import invoke_structured
from policy_expr.config import resolve_llm_config
from policy_expr.policies.base import resize_to_logical_png
from policy_expr.schemas import Observation, PolicyDecision
from policy_expr.validators.base import ValidationResult

load_dotenv()


SYSTEM_PROMPT = """\
你是一个 iPhone 自动化测试验证器。
你会看到动作前截图、动作后截图，以及本轮策略输出的动作。
请判断：动作后截图是否体现出该动作已经按预期生效。

判断标准：
- 只验证这一轮动作，不要求完整完成用户最终目标。
- tap：目标界面、弹层、焦点、页面变化等是否符合点击说明。
- type：目标输入框中是否出现了要输入的文字，或搜索/提交结果是否合理出现。
- scroll：屏幕内容是否发生了符合方向的滚动变化。
- home：是否回到主屏幕。
- 如果无法从截图确认，passed=false，并说明原因。

如果提供了用户目标（goal），还需额外判断：动作后截图是否已完整达成该目标。
- goal_completed=true：目标已完全实现，无需再进行任何操作。
- goal_completed=false：目标尚未完成，仍需继续。
- 未提供 goal 时，goal_completed 填 null。
请将判断结果填入 goal_completed 和 goal_completed_reason。

请用中文填写所有描述性字段。
"""


class SimpleLLMValidator:
    """General-purpose before/after screenshot validator."""

    name = "simple"

    def validate(
        self,
        before: Observation,
        decision: PolicyDecision,
        after: Observation,
        goal: str = "",
    ) -> ValidationResult:
        cfg = resolve_llm_config("validator")
        print(f"Validator Provider : {cfg.provider}")
        print(f"Validator Model    : {cfg.model}")

        llm = ChatOpenAI(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )

        action = decision.action
        before_b64 = _image_b64(before.png_bytes)
        after_b64 = _image_b64(after.png_bytes)
        goal_line = f"用户目标（goal）：{goal}\n\n" if goal else ""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"{goal_line}"
                            "本轮动作：\n"
                            f"- action_type: {action.action_type}\n"
                            f"- description: {action.description}\n"
                            f"- x/y: {action.x}, {action.y}\n"
                            f"- direction: {action.direction}\n"
                            f"- text: {action.text!r}\n\n"
                            f"策略屏幕理解：{decision.summary}\n"
                            f"策略推理：{decision.reasoning}\n\n"
                            "第一张图是动作前，第二张图是动作后。请判断动作是否生效。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{before_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{after_b64}"},
                    },
                ]
            ),
        ]
        return invoke_structured(llm, messages, ValidationResult)


def _image_b64(png_bytes: bytes) -> str:
    return base64.b64encode(resize_to_logical_png(png_bytes)).decode()

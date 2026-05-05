"""Structured-output multimodal LLM action policy."""

import base64

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from policy_expr.config import resolve_llm_config
from llm.structured import invoke_structured
from policy_expr.policies.base import BaseActionPolicy, resize_to_logical_png
from policy_expr.schemas import ActionDecision, Observation

load_dotenv()


SYSTEM_PROMPT = """\
你是一个 iPhone 操作执行器。
用户会提供 iPhone 截图和一个具体的操作指令。你只需要找到目标元素并输出对应动作。

坐标使用归一化坐标系：左上角(0,0)，右下角(1000,1000)。
需要在输入框中输入文字时，使用 type（而非 tap），它会自动先点击再输入。
需要滚动页面时，使用 scroll，只填写 direction（up=向上滚动看更多，down=向下滚动回顶部）。
需要返回主屏幕时，使用 home，无需填写坐标。
action 的 description 用中文简要说明操作目标即可。
"""


class StructuredOutputPolicy(BaseActionPolicy):
    """Vision-based action policy: LLM screenshot analysis + structured output."""

    name = "structured_output"

    def decide(self, observation: Observation, instruction: str) -> ActionDecision:
        cfg = resolve_llm_config("action_policy")
        print(f"Provider : {cfg.provider}")
        print(f"Model    : {cfg.model}")

        b64 = base64.b64encode(resize_to_logical_png(observation.png_bytes)).decode()
        llm = ChatOpenAI(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"操作指令：{instruction}\n\n请根据截图执行该指令。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ]
            ),
        ]
        return invoke_structured(llm, messages, ActionDecision)

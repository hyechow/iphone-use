"""Structured-output multimodal LLM policy."""

import base64

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from llm.provider_config import resolve_chat_provider_config
from policy_expr.policies.base import resize_to_logical_png
from policy_expr.schemas import Observation, PolicyDecision

load_dotenv()


SYSTEM_PROMPT = """\
你是一个专业的 iPhone 操作助手。
用户会提供 iPhone 截图和一个目标指令。请简短思考：
- 当前界面是什么状态？
- 为了完成目标，现在最优先的下一步操作是什么？
- 该操作的目标元素在哪里，坐标怎么判断？

只关注"现在要做什么"，不要列出后续所有步骤。
坐标使用归一化坐标系：左上角(0,0)，右下角(1000,1000)。
需要在输入框中输入文字时，使用 type（而非 tap），它会自动先点击再输入。
需要滚动页面时，使用 scroll，填写滚动区域中心坐标和 direction（up=向上滚动看更多，down=向下滚动回顶部）。
需要返回主屏幕时，使用 home，无需填写坐标。
将简短思考写入 reasoning，再据此填写 action。
请用中文填写所有描述性字段。
"""


class StructuredOutputPolicy:
    """Current baseline policy: LLM vision + structured output."""

    name = "structured_output"

    def decide(self, observation: Observation, prompt: str) -> PolicyDecision:
        cfg = resolve_chat_provider_config()
        print(f"Provider : {cfg.provider}")
        print(f"Model    : {cfg.model}")

        b64 = base64.b64encode(resize_to_logical_png(observation.png_bytes)).decode()
        llm = ChatOpenAI(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        ).with_structured_output(PolicyDecision)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"目标：{prompt}\n\n请根据截图判断当前应该执行哪些操作。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ]
            ),
        ]
        return llm.invoke(messages)

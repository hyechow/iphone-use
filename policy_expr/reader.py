"""Content Reader: extract relevant information from a screenshot during browsing."""

import base64

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from policy_expr.config import resolve_llm_config
from policy_expr.policies.base import resize_to_logical_png

load_dotenv()

SYSTEM_PROMPT = """\
你是一个手机屏幕内容提取助手。用户正在浏览手机上的某个页面，你的任务是从当前截图中提取与用户目标相关的关键内容。

要求：
- 只提取与目标问题相关的信息，忽略无关的 UI 元素（按钮、导航栏等）。
- 提取真实可见的文字内容，不要推测或补充截图中没有的信息。
- 简洁，几句话或几个要点，不要超过 150 字。
- 如果当前截图没有与目标相关的内容，回复"无相关内容"。
"""


class ContentReader:
    """Extract content notes from a screenshot relevant to the user's goal."""

    def __init__(self) -> None:
        cfg = resolve_llm_config("reader")
        self._llm = ChatOpenAI(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)

    def read(self, png_bytes: bytes, goal: str) -> str:
        """Return a brief content summary extracted from the screenshot."""
        b64 = base64.b64encode(resize_to_logical_png(png_bytes)).decode()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": f"用户目标：{goal}\n\n请提取当前截图中与目标相关的内容："},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]),
        ]
        response = self._llm.invoke(messages)
        text = response.content if isinstance(response.content, str) else str(response.content)
        return text.strip()

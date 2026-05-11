"""Content Reader: extract relevant information from a screenshot during browsing."""

import base64
import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from policy_expr.config import resolve_llm_config
from policy_expr.policies.base import resize_to_logical_png
from policy_expr.schemas import CollectionScope, SupervisorStep

load_dotenv()

SYSTEM_PROMPT = """\
从截图中提取与目标相关的数据。格式要求：
- 每条记录一行，字段用|分隔，保留与目标相关的字段（时间、名称、数值、状态等）
- 页面汇总数字只在覆盖范围与目标完全匹配时才提取
- 不解释，不汇总，不加多余文字
- 不评论列表是否到底、是否还有更多内容，只管提取当前可见数据
- 最多200字
- 无相关内容则回复"无相关内容"
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


def build_reader_instruction(original_goal: str, sv_step: SupervisorStep) -> str:
    """Build the extraction prompt for ContentReader based on milestone kind."""
    instruction = sv_step.read_instruction or original_goal
    if sv_step.milestone_kind != "collection":
        return instruction
    return (
        f"目标：{original_goal}\n"
        f"采集要求：{instruction}\n"
        "逐条提取当前屏幕可见记录，每条一行。记录可见范围/边界。"
    )


def annotate_content_note(
    note: str,
    *,
    turn_no: int,
    sv_step: SupervisorStep,
    collection_scope: CollectionScope | None,
) -> str:
    """Prepend collection metadata to a content note for traceability."""
    if sv_step.milestone_kind != "collection":
        return note
    metadata = [f"[turn{turn_no} {sv_step.milestone_id or '?'}]"]
    if collection_scope:
        metadata.append(
            "范围:" + json.dumps(collection_scope.model_dump(exclude_none=True), ensure_ascii=False)
        )
    return " ".join(metadata) + "\n" + note

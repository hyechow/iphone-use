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
你是一个手机屏幕内容提取助手。用户正在浏览手机上的某个页面，你的任务是从当前截图中提取与用户目标相关的关键内容。

要求：
- 只提取与目标问题相关的信息，忽略无关的 UI 元素（按钮、导航栏等）。
- 提取真实可见的文字内容，不要推测或补充截图中没有的信息。
- 列表、记录流、搜索结果、消息流等滚动内容必须逐条提取，不要只做汇总。
- 每条尽量保留与用户目标相关的可见字段，如时间/日期、名称/标题、数值、状态、类别、来源、备注等；截图看不到的字段写"未显示"。
- 如果截图上可见筛选范围、分组标题、排序条件或结果范围，先单独写一行"可见范围/条件：..."。
- 不要计算跨截图总和，不要把当前截图内容说成全部内容，除非截图明确显示已到达列表边界。
- 页面顶部或标题区域显示的汇总统计数字（如总支出、总收入、账单合计等），只有当其覆盖范围与用户目标时间范围完全匹配时才提取；若覆盖范围更宽（如目标是"本周"但汇总是"本月"），跳过该数字，只提取逐条明细记录。
- 简洁但完整，最多 600 字。
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


def build_reader_instruction(original_goal: str, sv_step: SupervisorStep) -> str:
    """Build the extraction prompt for ContentReader based on milestone kind."""
    instruction = sv_step.read_instruction or original_goal
    if sv_step.milestone_kind != "collection":
        return instruction
    return (
        f"用户最终目标：{original_goal}\n"
        f"本轮采集要求：{instruction}\n\n"
        "请逐条提取当前屏幕可见的相关记录，输出结构化文本。"
        "列表/记录类内容每条至少包含：可见时间或位置、名称或标题、目标相关数值、状态、类别或备注；"
        "如果字段不可见写「未显示」。"
        "如果屏幕显示筛选范围、分组条件、排序条件或列表边界，也必须记录。"
        "不要只输出单一字段列表，不要跨截图合计，不要把当前屏幕说成全部内容。"
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
    metadata = [
        f"采集轮次：turn {turn_no}",
        f"子目标：{sv_step.milestone_id or '未知'}",
    ]
    if collection_scope:
        metadata.append(
            "采集范围："
            + json.dumps(collection_scope.model_dump(exclude_none=True), ensure_ascii=False)
        )
    metadata.append("说明：以下内容来自同一筛选/分组条件下的当前可见列表，列表滚动时相邻屏可能存在重叠。")
    return "\n".join(metadata) + "\n" + note

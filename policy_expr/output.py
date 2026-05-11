"""Final user-facing summaries for policy experiment runs."""

import json
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from llm.structured import invoke_structured
from policy_expr.config import resolve_llm_config
from policy_expr.schemas import PolicyTurn

load_dotenv()


ACTION_SYSTEM_PROMPT = """\
你是 iPhone 自动化任务的最终结果总结助手。
你会收到一次策略运行的完整 context，包括用户目标、停止原因、每轮观察、动作、执行状态和验证结果。
请基于这些事实判断任务最终状态，并用中文输出给用户看的简短摘要。

要求：
- 不要输出详细 Markdown 报告，不要逐轮罗列日志。
- 控制在 3-6 句话。
- 必须说明任务是否已完成、关键依据。
- 如果 context 无法确认完成，不要猜测，明确说"未确认"或"未完成"。
- 不要提及停止原因、运行模式、日志目录或日志保存位置。
- 不要在结尾追加"任务因...停止""完整日志保存在..."之类的运行说明。
"""

ANALYSIS_SYSTEM_PROMPT = """\
你是 iPhone 信息收集任务的最终结果整理助手。
用户让 agent 在手机上浏览并收集信息，agent 已逐页提取了相关内容片段。
你的任务是将这些片段整合成一份直接回答用户目标的简洁报告。

要求：
- 直接呈现收集到的信息内容，不要描述 agent 的操作过程。
- 合并重复内容，保留关键细节。
- 如果信息不完整，如实说明，不要补充截图中没有的内容。
- **数据校验**：如果用户目标包含特定条件（如范围、实体、类别、状态、排序、数量或数值条件），必须检查收集到的数据是否满足这些条件。如果数据与目标条件不匹配，必须明确指出不匹配点，不要将错误范围或错误条件下的数据当作正确答案输出。
- 如果运行结论说明"未完成"或"数据校验不充分"，必须先明确说明当前数据不足，不能把不充分的数据包装成确定答案。
- 不要提及"agent"、"截图"、"收集"等操作性词汇，直接给出答案。
- 不要在结尾追加运行说明。
"""


def render_final_output(
    goal: str,
    policy_name: str,
    turns: Sequence[PolicyTurn],
    log_dir: Path,
    stop_reason: str,
    content_notes: list[str] | None = None,
    collection_context: str | None = None,
) -> str:
    """Use an LLM to render a concise final summary for a finished policy run."""

    cfg = resolve_llm_config("output")
    llm = ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )

    if content_notes:
        notes_text = "\n\n".join(f"[片段 {i+1}]\n{note}" for i, note in enumerate(content_notes))
        if collection_context:
            notes_text = f"[采集上下文] {collection_context}\n\n以下为逐帧提取的内容片段：\n\n{notes_text}"
        messages = [
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"用户目标：{goal}\n\n"
                    f"运行结论：{stop_reason}\n\n"
                    f"收集到的内容片段：\n{notes_text}"
                )
            ),
        ]
    else:
        context = _build_output_context(goal, policy_name, turns, log_dir, stop_reason)
        messages = [
            SystemMessage(content=ACTION_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "请根据以下完整运行 context 生成最终摘要：\n"
                    f"{json.dumps(context, ensure_ascii=False, indent=2)}"
                )
            ),
        ]

    response = llm.invoke(messages)
    return _message_text(response.content).strip() + "\n"


def _build_output_context(
    goal: str,
    policy_name: str,
    turns: Sequence[PolicyTurn],
    log_dir: Path,
    stop_reason: str,
) -> dict:
    return {
        "goal": goal or "未提供",
        "policy_name": policy_name,
        "stop_reason": stop_reason,
        "turn_count": len(turns),
        "log_dir": str(log_dir),
        "turns": [turn.model_dump(mode="json") for turn in turns],
    }


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)

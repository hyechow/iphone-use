"""Build page operation knowledge from a single recon result.

Each recon directory represents ONE page. This module extracts that page's
knowledge (description + elements + what each element does) for use as LLM
context. Target pages from taps are NOT included — they are future exploration
targets.

Raw elements from recon are abstracted by LLM: similar items merged, private
info removed, producing concise general-purpose operations.

Output is a skill: brief description for matching + operations for execution.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm.structured import invoke_structured
from policy_expr.config import resolve_llm_config


# ── Data models ──────────────────────────────────────────

class ElementKnowledge(BaseModel):
    """One interactive element on a page (post-abstraction)."""
    label: str
    position: str = Field(description="语义位置，如「右上角」「底部导航栏」「列表区域」")
    function: str = Field(description="功能描述，如「进入搜索页面」「弹出操作菜单」")


class PageKnowledge(BaseModel):
    """Knowledge about one page, formatted as a skill for LLM injection."""
    app_name: str
    signature: str
    page_name: str
    description: str
    operations: list[ElementKnowledge]

    def to_skill(self) -> str:
        """Generate skill-format text with frontmatter metadata.

        Frontmatter (system use): signature, app for indexing/matching.
        Body (LLM context): description + operations.
        """
        lines = [
            "---",
            f"app: {self.app_name}",
            f"signature: {self.signature}",
            "---",
            "",
            f"# {self.page_name}",
            "",
            self.description,
        ]
        if self.operations:
            lines.append("")
            for op in self.operations:
                lines.append(f"- [{op.label}] {op.position} → {op.function}")
        return "\n".join(lines)


# ── Abstraction ──────────────────────────────────────────

ABSTRACT_PROMPT = """\
你是一个 iPhone 应用页面分析专家。你会看到一个页面的简要描述和所有可交互元素
的原始列表。请完成两个任务：

## 任务一：页面描述（1-2句话，80字以内）
用简洁的一句话或两句话概括这个页面的核心功能和内容。
要求：
- 通用化，不包含具体的联系人、消息内容等私有信息
- 说明页面用途和包含的关键功能区
- LLM 只看这段描述就能判断页面是否与当前任务相关

## 任务二：抽象操作列表
将原始元素抽象总结为通用操作列表。要求：
1. **合并同类项**：功能相同的元素合并为一条
2. **去除私有信息**：不包含具体联系人名称、手机号、消息内容
3. **通用化描述**：适用于该页面的任意状态
4. **保留结构信息**：保留位置和功能描述
"""


class AbstractedOperation(BaseModel):
    """LLM output: one abstracted operation."""
    label: str = Field(description="操作的通用名称，如「聊天行」「+号按钮」「搜索栏」")
    position: str = Field(description="语义位置，如「右上角」「底部导航栏」「列表区域」")
    function: str = Field(description="功能描述，如「进入该联系人的聊天详情页」「弹出操作菜单」")


class AbstractedPage(BaseModel):
    """LLM output: abstracted page description + operations."""
    description: str = Field(description="页面描述，1-2句话，80字以内")
    operations: list[AbstractedOperation]


def _abstract_page(brief_desc: str, raw_elements: list[str]) -> AbstractedPage:
    """Ask LLM to generate description + abstract operations."""
    cfg = resolve_llm_config("action_policy")
    llm = ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=0,
    )

    element_text = "\n".join(f"  {e}" for e in raw_elements)
    messages = [
        SystemMessage(content=ABSTRACT_PROMPT),
        HumanMessage(content=(
            f"页面简要描述：{brief_desc}\n\n"
            f"原始元素列表：\n{element_text}"
        )),
    ]

    return invoke_structured(llm, messages, AbstractedPage)


# ── Helpers ───────────────────────────────────────────────

def _semantic_position(x: float, y: float, element_type: str) -> str:
    """Map normalized coordinates + element type to a semantic position."""
    if element_type == "tab":
        return "底部导航栏"
    if element_type == "back_button":
        return "左上角"
    if y < 180:
        return "右上角" if x > 700 else "顶部"
    if y > 900:
        return "底部"
    if element_type == "input":
        return "输入区域"
    return "列表区域"


# ── Public API ────────────────────────────────────────────

def build_knowledge(recon_dir: Path) -> PageKnowledge:
    """Build page knowledge from a single recon result.

    1. Read raw elements from recon outputs
    2. Abstract via LLM: merge similar, remove private info
    3. Return structured PageKnowledge

    Args:
        recon_dir: Directory containing recon outputs.
    """
    recon_dir = recon_dir.resolve()

    recon = json.loads((recon_dir / "recon_result.json").read_text("utf-8"))
    flows_data = json.loads((recon_dir / "page_flows.json").read_text("utf-8"))
    initial_result = json.loads((recon_dir / "initial_result.json").read_text("utf-8"))

    page_info = initial_result["page"]
    taps = recon["taps"]
    flows = flows_data["flows"]

    # Build raw element descriptions for abstraction
    raw_elements = []
    for tap, flow in zip(taps, flows):
        el_type = tap["element_type"]
        pos = _semantic_position(tap["x"], tap["y"], el_type)
        raw_elements.append(f"[{tap['label']}]  {pos}  → {flow['flow_description']}")

    print(f"  原始元素: {len(raw_elements)} 个，正在抽象...")
    abstracted = _abstract_page(page_info["description"], raw_elements)
    print(f"  抽象后: {len(abstracted.operations)} 个操作")

    return PageKnowledge(
        app_name=recon["app_name"],
        signature=page_info["identity"]["signature"],
        page_name=page_info["identity"]["page_title"],
        description=abstracted.description,
        operations=[
            ElementKnowledge(label=op.label, position=op.position, function=op.function)
            for op in abstracted.operations
        ],
    )


def save_knowledge(kb: PageKnowledge, out_path: Path) -> None:
    """Save page knowledge as markdown."""
    out_path.write_text(kb.to_skill(), encoding="utf-8")

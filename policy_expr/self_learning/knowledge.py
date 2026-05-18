"""Export page knowledge from recon results.

One LLM call per page: reads initial_result.json + recon_result.json,
produces page_meta.json (structured identity) + knowledge.md (skill text).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm.structured import invoke_structured
from policy_expr.config import resolve_llm_config


PageType = Literal["list", "detail", "chat", "form", "modal", "home", "other"]


# ── Output models ─────────────────────────────────────────

class ElementKnowledge(BaseModel):
    label: str
    position: str = Field(description="语义位置，如「右上角」「底部导航栏」「列表区域」")
    function: str = Field(description="功能描述，如「进入搜索页面」「弹出操作菜单」")


class PageKnowledge(BaseModel):
    """LLM output: normalized page knowledge."""
    page_title: str = Field(description="2-4个字的页面标题，如「聊天列表」「个人主页」")
    page_type: PageType = Field(description="页面类型")
    description: str = Field(description="1-2句话概括页面功能，通用化，不含私有信息")
    operations: list[ElementKnowledge]

    def to_skill(self, app: str, parent_page: str = "") -> str:
        lines = [
            "---",
            f"app: {app}",
            f"page_title: {self.page_title}",
            f"page_type: {self.page_type}",
        ]
        if parent_page:
            lines.append(f"parent_page: {parent_page}")
        lines += ["---", "", f"# {self.page_title}", "", self.description]
        if self.operations:
            lines.append("")
            for op in self.operations:
                lines.append(f"- [{op.label}] {op.position} → {op.function}")
        return "\n".join(lines)


@dataclass
class PageMeta:
    page_title: str
    page_type: str
    parent_page: str
    description: str

    def to_dict(self) -> dict:
        return {
            "page_title": self.page_title,
            "page_type": self.page_type,
            "parent_page": self.parent_page,
            "description": self.description,
        }


@dataclass
class ExportResult:
    meta: PageMeta
    knowledge: PageKnowledge


# ── LLM prompt ───────────────────────────────────────────

EXPORT_PROMPT = """\
你是一个 iPhone 应用页面分析专家。给定一个页面的探测数据，完成以下任务：

## 输出要求

**page_title**：页面的唯一标识名称，4-8个字。要求：
- 必须包含功能域 + 页面形态，如「公众号订阅列表」「群聊消息详情」「联系人个人资料」
- 禁止使用纯通用词（「列表页」「详情页」「主页」），必须加上具体功能域
- 同一应用内每个页面的 page_title 必须互不相同，可区分

**page_type**：页面类型
- list：列表页（聊天列表、联系人、消息列表）
- detail：详情页（个人资料、文章详情）
- chat：聊天/对话界面
- form：表单/输入页
- modal：弹窗/底部弹出
- home：应用主页
- other：其他

**description**：1-2句话概括页面功能。要求：
- 通用化，不含具体联系人、消息内容等私有信息
- 说明页面用途和关键功能区

**operations**：抽象操作列表。核心原则：**label 必须是通用名称，绝对不能出现具体内容**。

抽象规则：
1. **列表行合并**：同类型的多行（聊天、文章、联系人、商品等）合并为一条，label 用通用名如「聊天行」「文章行」「联系人行」
2. **去除所有具体内容**：联系人名、文章标题、消息正文、账号名、金额等一律替换为通用描述
3. **保留功能性标签**：搜索栏、+号按钮、设置按钮等本身就是通用名称，保持原样
4. **function 描述实测结果**：有导航实测的标注「进入…」，无导航的描述其交互用途

示例（错误 → 正确）：
- ✗ [Mythos 限] → ✓ [文章行]，function：进入文章详情页
- ✗ [张三] → ✓ [聊天行]，function：进入该联系人的聊天详情
- ✗ [2024年终总结] → ✓ [文章行]，function：查看文章内容
"""


def _semantic_position(x: float, y: float, element_type: str) -> str:
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


def _build_element_lines(taps: list[dict]) -> list[str]:
    """Build element description lines from probe taps only (verified data)."""
    lines = []
    for tap in taps:
        label = tap.get("label", "")
        el_type = tap.get("element_type", "area")
        x, y = tap.get("x", 500), tap.get("y", 500)
        pos = _semantic_position(x, y, el_type)

        if tap.get("navigated"):
            dest = (tap.get("identity") or {}).get("description", "")
            nav_note = f"实测→「{dest[:30]}」" if dest else "实测→已导航"
            lines.append(f"[{label}]  {pos}  {nav_note}")
        else:
            lines.append(f"[{label}]  {pos}  无导航")

    return lines


# ── Public API ────────────────────────────────────────────

def build_export(page_dir: Path) -> ExportResult:
    """Build export for one page directory.

    Reads initial_result.json + recon_result.json, calls LLM once,
    returns ExportResult containing PageMeta + PageKnowledge.
    """
    page_dir = page_dir.resolve()

    init_path = page_dir / "initial_result.json"
    recon_path = page_dir / "recon_result.json"

    if not init_path.exists():
        raise FileNotFoundError(f"initial_result.json not found: {page_dir}")
    if not recon_path.exists():
        raise FileNotFoundError(f"recon_result.json not found: {page_dir}")

    init_data = json.loads(init_path.read_text("utf-8"))
    recon_data = json.loads(recon_path.read_text("utf-8"))

    raw_description = init_data.get("page", {}).get("description", "")
    taps: list[dict] = recon_data.get("taps", [])
    parent_page: str = recon_data.get("parent_page", "")

    element_lines = _build_element_lines(taps)

    cfg = resolve_llm_config("action_policy")
    llm = ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=0,
    )

    element_text = "\n".join(f"  {l}" for l in element_lines)
    messages = [
        SystemMessage(content=EXPORT_PROMPT),
        HumanMessage(content=(
            f"父页面：{parent_page or '无（根页面）'}\n"
            f"页面描述：{raw_description}\n\n"
            f"可交互元素（{len(element_lines)} 个）：\n{element_text}"
        )),
    ]

    knowledge = invoke_structured(llm, messages, PageKnowledge)

    meta = PageMeta(
        page_title=knowledge.page_title,
        page_type=knowledge.page_type,
        parent_page=parent_page,
        description=knowledge.description,
    )

    return ExportResult(meta=meta, knowledge=knowledge)


def save_export(result: ExportResult, page_dir: Path, knowledge_dir: Path) -> None:
    """Save page_meta.json locally and knowledge.md to knowledge_dir."""
    # page_meta.json in the recon page directory
    meta_path = page_dir / "page_meta.json"
    meta_path.write_text(
        json.dumps(result.meta.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Derive app name from knowledge_dir name
    app = knowledge_dir.name

    # knowledge.md: local copy + sync to knowledge/{app}/
    skill_text = result.knowledge.to_skill(app, result.meta.parent_page)
    (page_dir / "knowledge.md").write_text(skill_text, encoding="utf-8")

    knowledge_dir.mkdir(parents=True, exist_ok=True)
    safe_title = result.meta.page_title.replace("/", "_").replace(" ", "_")
    (knowledge_dir / f"{safe_title}.md").write_text(skill_text, encoding="utf-8")

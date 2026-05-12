"""Page flow description: recon result → structured functional knowledge.

Takes a ReconResult (initial screenshot + tap screenshots) and generates
a functional flow description for each navigated tap via vision LLM.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm.structured import invoke_structured
from policy_expr.config import resolve_llm_config
from policy_expr.policies.base import resize_to_logical_png


# ── Data models ──────────────────────────────────────────

PageType = Literal[
    "list", "detail", "form", "dialog", "home", "settings", "webview", "other",
]


class FlowTransition(BaseModel):
    """One page transition learned from tapping an element."""
    target_page: str = Field(description="目标页面名称，如「新建对话」「搜索页」「聊天详情」")
    target_description: str = Field(description="目标页面的一句话功能描述")
    target_page_type: PageType = Field(description="目标页面类型")
    flow_description: str = Field(
        description="完整的功能描述，格式：「在[源页面]，点击「元素」→ [效果]」"
    )


class PageFlow(BaseModel):
    """Functional flow knowledge for one page."""
    app_name: str
    page_title: str
    page_description: str
    flows: list[FlowTransition] = Field(default_factory=list)


# ── LLM prompt ───────────────────────────────────────────

TRANSITION_SYSTEM_PROMPT = """\
你是一个 iPhone 应用界面分析专家。你会看到两张截图：
- 图1（源页面）：用户当前所在的页面
- 图2（目标页面）：点击某个元素后进入的页面

请分析目标页面，描述这个页面跳转的功能含义。

输出字段说明：
- target_page: 目标页面的名称
- target_description: 目标页面的功能描述（一句话）
- target_page_type: 目标页面类型（list/detail/form/dialog/home/settings/webview/other）
- flow_description: 完整功能描述，格式："在[源页面]，点击「元素」→ [效果]"
"""


# ── Core logic ───────────────────────────────────────────

def describe_transition(
    source_b64: str,
    target_b64: str,
    source_desc: str,
    element_label: str,
) -> FlowTransition:
    """Ask vision LLM to describe a page transition."""
    cfg = resolve_llm_config("action_policy")
    llm = ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=0,
    )

    user_text = (
        f"源页面描述：{source_desc}\n"
        f"点击的元素：「{element_label}」\n\n"
        "请分析目标页面（图2）并描述这个页面跳转。"
    )

    messages = [
        SystemMessage(content=TRANSITION_SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": "图1（源页面）："},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{source_b64}"}},
            {"type": "text", "text": "\n图2（目标页面）："},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_b64}"}},
            {"type": "text", "text": f"\n{user_text}"},
        ]),
    ]

    return invoke_structured(llm, messages, FlowTransition)


def _resolve_screenshot(path: Path, fallback: str, result_dir: Path) -> Path | None:
    """Resolve a screenshot path relative to result_dir."""
    if path.exists():
        return path
    resolved = result_dir / path.name
    if resolved.exists():
        return resolved
    fb = Path(fallback)
    if fb.exists():
        return fb
    return None


def _load_b64(path: Path) -> str:
    """Read a PNG file and return base64 of its logical-resolution version."""
    png = path.read_bytes()
    return base64.b64encode(resize_to_logical_png(png)).decode()


def build_page_flows(recon_path: Path) -> PageFlow:
    """Build functional flow descriptions from a recon result.

    Args:
        recon_path: Path to recon_result.json.

    Returns:
        PageFlow with flow descriptions for each navigated tap.
    """
    recon_path = recon_path.resolve()
    result_dir = recon_path.parent
    recon = json.loads(recon_path.read_text(encoding="utf-8"))

    # Resolve initial screenshot
    initial_path = _resolve_screenshot(
        result_dir / "initial.png",
        recon.get("initial_screenshot", ""),
        result_dir,
    )
    if not initial_path:
        raise FileNotFoundError(f"Initial screenshot not found in {result_dir}")

    initial_b64 = _load_b64(initial_path)
    source_desc = recon["description"]

    # Process all taps (tabs navigate to different pages too)
    flows: list[FlowTransition] = []

    for tap in recon["taps"]:
        tap_filename = Path(tap["screenshot"]).name
        tap_path = _resolve_screenshot(
            result_dir / "tap" / tap_filename,
            tap["screenshot"],
            result_dir,
        )
        if not tap_path:
            print(f"  [SKIP] 截图不存在: {tap['screenshot']}")
            continue

        label = tap["label"]
        print(f"  [{tap['index']:2d}] 点击「{label}」...")

        target_b64 = _load_b64(tap_path)
        transition = describe_transition(initial_b64, target_b64, source_desc, label)
        flows.append(transition)
        print(f"       → {transition.flow_description}")

    return PageFlow(
        app_name=recon["app_name"],
        page_title=recon["page_title"],
        page_description=source_desc,
        flows=flows,
    )


def save_page_flows(page_flow: PageFlow, out_path: Path) -> None:
    """Save PageFlow to JSON."""
    out_path.write_text(
        page_flow.model_dump_json(indent=2),
        encoding="utf-8",
    )

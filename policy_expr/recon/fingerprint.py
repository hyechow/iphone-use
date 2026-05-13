"""Page fingerprint: deterministic layout signature for page dedup."""

from __future__ import annotations

import base64
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm.structured import invoke_structured
from policy_expr.config import resolve_llm_config
from policy_expr.policies.base import resize_to_logical_png


# ── Enums ──────────────────────────────────────────────────

PageType = Literal[
    "list", "detail", "chat", "home", "settings",
    "form", "dialog", "search", "webview",
]

TopBar = Literal[
    "none", "search_bar", "back_title", "back_title_more",
    "title_icons", "profile_header",
]

ContentArea = Literal[
    "chat_bubbles", "avatar_rows", "card_grid", "text_lines",
    "feed", "form", "mixed",
]

BottomBar = Literal[
    "none", "tabs", "input", "buttons",
]


# ── Model ──────────────────────────────────────────────────

class PageFingerprint(BaseModel):
    page_type: PageType
    top_bar: TopBar
    content_area: ContentArea
    bottom_bar: BottomBar
    has_bottom_tabs: bool = Field(description="底部是否有 tab 切换栏")
    tab_count: int = Field(default=0, ge=0, le=10)

    @property
    def key(self) -> str:
        return (
            f"{self.page_type}|{self.top_bar}|{self.content_area}"
            f"|{self.bottom_bar}|tabs={self.tab_count}"
        )


# ── Prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = """\
分析手机页面的布局骨架，忽略所有文字内容和动态信息。

## 顶部导航
- none: 没有导航栏
- search_bar: 搜索框（可附带分类入口）
- back_title: 返回按钮 + 标题
- back_title_more: 返回按钮 + 标题 + 右侧图标
- title_icons: 标题 + 右侧图标（无返回按钮）
- profile_header: 头像 + 昵称（个人中心页）

## 内容区
- chat_bubbles: 聊天气泡
- avatar_rows: 每行有头像/缩略图的列表
- card_grid: 卡片式布局（圆角块、商品卡片）
- text_lines: 纯文字行列表（无头像）
- feed: 信息流/瀑布流/混合内容滚动区
- form: 表单输入区域
- mixed: 多种内容混合的详情页

## 底部栏
- none: 无底部栏
- tabs: 只有 tab 切换栏（底部有一排等宽的图标+文字按钮）
- input: 含输入框的区域（含/不含附加按钮都选此项）
- buttons: 纯操作按钮行（无输入框、非 tab 切换）

## 判断底部 tab 栏的方法
底部 tab 栏的特征：一排等宽的图标按钮，每个下方有文字标签，用于在几个主要页面间切换。
数一下有几个 tab 图标，填入 tab_count。如果底部没有这种 tab 切换栏，has_bottom_tabs=false，tab_count=0。
"""


# ── API ────────────────────────────────────────────────────

def compute_fingerprint(png_bytes: bytes) -> PageFingerprint:
    """Compute layout fingerprint from a screenshot."""
    cfg = resolve_llm_config("action_policy")
    llm = ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=0,
    )
    b64 = base64.b64encode(resize_to_logical_png(png_bytes)).decode()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": "分类这个页面，特别注意底部是否有 tab 切换栏。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]),
    ]
    return invoke_structured(llm, messages, PageFingerprint)

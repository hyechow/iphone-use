"""Page parser: screenshot → page identity + interactive elements."""

from __future__ import annotations

import base64
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm.structured import invoke_structured
from policy_expr.config import resolve_llm_config
from policy_expr.policies.base import resize_to_logical_png


ElementType = Literal[
    "back_button",  # < / 返回
    "tab",          # bottom / top tab bar item
    "button",       # tappable control (non-nav)
    "link",         # row / cell that navigates to another screen
    "input",        # text field / search bar
    "menu_item",    # action sheet / context menu item
    "icon",         # icon-only tappable with no visible label
]

# Semantic category for icon-only elements (element_type == "icon")
IconSemantic = Literal[
    "search",        # 放大镜
    "settings",      # 齿轮/设置
    "close",         # × 关闭
    "share",         # 分享/上传箭头
    "more",          # … / ⋮ 更多选项
    "notification",  # 铃铛/通知
    "profile",       # 人像/头像
    "camera",        # 相机
    "scan",          # 扫码/二维码
    "add",           # + 新增
    "edit",          # 铅笔/编辑
    "delete",        # 垃圾桶/删除
    "favorite",      # 心形/星形/收藏
    "filter",        # 漏斗/筛选
    "download",      # 下载箭头
    "message",       # 消息气泡
    "map",           # 地图/位置定位
    "other",         # 无法归入以上类别
]


class InteractiveElement(BaseModel):
    label: str = Field(description="可见文字标签；无文字的纯图标元素填空字符串 ''")
    element_type: ElementType = Field(description="元素类型")
    icon_semantic: IconSemantic | None = Field(
        default=None,
        description=(
            "仅当 element_type='icon' 时必填，其余留 null。"
            "从常见 iOS 图标语义中选择最匹配的分类，"
            "如放大镜→search、齿轮→settings、×→close、三点→more 等"
        ),
    )
    x: float = Field(description="归一化 x 坐标（0-1000，左上角为原点）")
    y: float = Field(description="归一化 y 坐标（0-1000）")
    leads_to: str = Field(
        description="点击后的预期效果或目标页面，如「打开聊天详情」「展开搜索框」「返回上一页」"
    )


class PageIdentity(BaseModel):
    app_name: str = Field(description="当前前台应用名称，如「微信」「支付宝」")
    page_title: str = Field(description="页面标题或名称，如「聊天列表」「通讯录」「个人主页」")
    page_type: str = Field(
        description="页面类型，如 list / detail / home / settings / form / dialog / webview"
    )
    signature: str = Field(
        description=(
            "页面去重指纹，格式：「应用/页面/稳定特征」，"
            "只含稳定 UI 骨架，不含动态内容（未读数、时间戳、用户名等）。"
            "示例：「微信/聊天列表/tab[微信,通讯录,发现,我]」"
            "「微信/通讯录/section[新朋友,群聊,标签]」"
        )
    )


class ParsedPage(BaseModel):
    identity: PageIdentity
    description: str = Field(description="一句话描述该页面的功能或用途")
    interactive_elements: list[InteractiveElement] = Field(
        description="页面上所有可点击/可交互的元素，按从上到下、从左到右排列"
    )


SYSTEM_PROMPT = """\
你是一个 iPhone 页面分析器。仔细分析截图，输出页面身份和所有可交互元素。

坐标系：左上角 (0, 0)，右下角 (1000, 1000)。坐标代表元素的视觉中心。

## 扫描方法（严格遵守）
从上到下逐行扫描截图。对页面上每个可点击/可交互的区域，输出一条记录。
相邻的不同可交互元素必须分开输出，不要合并。

具体规则：
1. **导航栏**：返回按钮（<）、右侧图标按钮（分享、更多等），每个单独一条
2. **时间线/进度条**：每个节点（如"26日已签收"）单独一条 link
3. **地址栏**：「修改」链接、复制/拨号等小图标，每个单独一条
4. **商品区**：
   - 商品图片/缩略图 → 单独一条 link
   - 商品标题文字 → 单独一条 link
   - 店铺名称 → 单独一条 link
   - 服务标签（坏了包赔、免费送货上门等）→ 每个标签单独一条 link，即使它们在同一行
5. **操作按钮行**：分享、联系、退款等，每个单独一条 button
6. **推荐/促销区**：每个商品卡片、横幅、广告单独一条 link（如果有两张并排卡片就输出两条）
7. **底部操作栏**：每个 tab 和按钮单独一条

## 绝对不要做的事
- 不要把多个相邻的标签合并为一个元素（如"坏了包赔"和"免费送货上门"必须分开）
- 不要跳过商品图片/缩略图
- 不要跳过时间线中的节点
- 不要跳过推荐区的每个商品卡片

## 元素类型
- back_button：返回 < 或 ‹
- tab：导航栏的每个 tab
- button：有明确边框/背景的控件
- link：跳转到其他页面的文字、行、卡片（带 > 的行、商品卡片、列表行）
- input：文本输入框
- menu_item：弹出菜单项
- icon：纯图标按钮（无可见文字）

## icon_semantic（element_type = "icon" 时必填）
search / settings / close / share / more / notification / profile /
camera / scan / add / edit / delete / favorite / filter / download / message / map / other
有文字标签的元素 icon_semantic 留 null。

## signature（去重指纹）
只含稳定 UI 骨架，不含动态内容。格式：「应用名/页面名/关键骨架特征」
"""


def enrich_with_icons(
    page: ParsedPage,
    icon_bboxes: list,   # list[IconBbox] — avoid circular import
    img_w: int,
    img_h: int,
) -> ParsedPage:
    """Merge LLM elements with YOLO bboxes via point-in-box matching.

    For each LLM element, check if its center point (converted to pixel
    coords) falls inside a YOLO bbox.  If so, replace the LLM coordinates
    with the YOLO bbox center (pixel-level precision).

    YOLO bboxes that matched no LLM element are appended as extra icon
    elements so no detection results are lost.
    """
    matched_boxes: set[int] = set()
    merged: list[InteractiveElement] = []

    for el in page.interactive_elements:
        px = el.x / 1000.0 * img_w
        py = el.y / 1000.0 * img_h
        matched = False
        for bi, bbox in enumerate(icon_bboxes):
            if bi in matched_boxes:
                continue
            if bbox.x1 <= px <= bbox.x2 and bbox.y1 <= py <= bbox.y2:
                merged.append(InteractiveElement(
                    label=el.label,
                    element_type=el.element_type,
                    icon_semantic=el.icon_semantic,
                    x=round(bbox.cx / img_w * 1000, 1),
                    y=round(bbox.cy / img_h * 1000, 1),
                    leads_to=el.leads_to,
                ))
                matched_boxes.add(bi)
                matched = True
                break
        if not matched:
            merged.append(el)

    # Append unmatched YOLO bboxes as extra icons
    for bi, bbox in enumerate(icon_bboxes):
        if bi in matched_boxes:
            continue
        merged.append(InteractiveElement(
            label="",
            element_type="icon",
            icon_semantic="other",
            x=round(bbox.cx / img_w * 1000, 1),
            y=round(bbox.cy / img_h * 1000, 1),
            leads_to="",
        ))

    return ParsedPage(
        identity=page.identity,
        description=page.description,
        interactive_elements=merged,
    )


class PageParser:
    """Parse a screenshot into structured page identity and interactive elements."""

    _N_ATTEMPTS = 2

    def parse(self, png_bytes: bytes) -> ParsedPage:
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
            HumanMessage(
                content=[
                    {"type": "text", "text": "请分析这张 iPhone 截图，输出页面身份和所有可交互元素。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
            ),
        ]

        best: ParsedPage | None = None
        last_error: Exception | None = None
        for attempt in range(self._N_ATTEMPTS):
            try:
                result = invoke_structured(llm, messages, ParsedPage)
                if best is None or len(result.interactive_elements) > len(best.interactive_elements):
                    best = result
            except Exception as exc:
                last_error = exc
                print(f"  attempt {attempt + 1} 失败: {type(exc).__name__}")

        if best is not None:
            return best
        raise last_error or RuntimeError("no results")

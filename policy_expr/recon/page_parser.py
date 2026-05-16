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


class BottomNav(BaseModel):
    """底部导航栏检测结果。"""
    has_nav: bool = Field(description="页面是否有底部导航栏（tab bar / 底部操作栏）")


class ParsedPage(BaseModel):
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
    is_miniprogram: bool = Field(
        default=False,
        description=(
            "是否是微信小程序或 App 内嵌 H5/WebView 页面。"
            "判断依据：右上角有胶囊按钮（pill 形，左圆含 ···，右圆含 ×）。"
            "普通 iOS 原生页面填 false。"
        ),
    )
    description: str = Field(description="一句话描述该页面的功能或用途")
    bottom_nav: BottomNav = Field(
        description="底部导航栏区域，无则 y_start=y_end=-1"
    )
    interactive_elements: list[InteractiveElement] = Field(
        description="页面上所有可点击/可交互的元素，按从上到下、从左到右排列"
    )


class InteractiveArea(BaseModel):
    label: str = Field(description="简短标签，2-6 字")
    target_page: str = Field(description="目标页面名称")
    description: str = Field(description="简要描述")
    center_xy: list[float] = Field(description="可交互区中心坐标 [x, y]，0-1000")
    element_type: str = ""  # dominant type from underlying elements (e.g. "back_button", "tab")


class PageKnowledge(BaseModel):
    page: ParsedPage
    areas: list[InteractiveArea]
    llm_page: ParsedPage | None = None
    yolo_boxes: list | None = None
    img_size: tuple[int, int] | None = None


def classify_elements(page: ParsedPage) -> list[InteractiveArea]:
    """Group elements into tappable areas by leads_to + y proximity.

    Uses the LLM-generated leads_to (semantic) combined with spatial
    proximity — deterministic, no extra LLM call.
    """
    els = page.interactive_elements
    if not els:
        return []

    # Sort by y, then x
    indexed = sorted(range(len(els)), key=lambda i: (els[i].y, els[i].x))

    # Cluster by y proximity, then split by leads_to within each cluster
    groups: list[list[int]] = []
    current: list[int] = []

    for idx in indexed:
        if not current:
            current.append(idx)
            continue

        # y proximity check
        min_y = min(els[i].y for i in current)
        max_y = max(els[i].y for i in current)
        el = els[idx]

        if abs(el.y - min_y) < 40 or abs(el.y - max_y) < 40:
            # Within y range — merge if leads_to compatible
            current_leads = {els[i].leads_to for i in current if els[i].leads_to}
            if not el.leads_to or not current_leads or el.leads_to in current_leads:
                current.append(idx)
                continue
            # Same row but different target — split
            groups.append(current)
            current = [idx]
        else:
            # Different row
            if current:
                groups.append(current)
            current = [idx]

    if current:
        groups.append(current)

    ICON_LABEL = {
        "search": "搜索", "settings": "设置", "close": "关闭",
        "share": "分享", "more": "更多", "notification": "通知",
        "profile": "头像", "camera": "相机", "scan": "扫码",
        "add": "新建", "edit": "编辑", "delete": "删除",
        "favorite": "收藏", "filter": "筛选", "download": "下载",
        "message": "消息", "map": "地图", "other": "图标",
    }

    # Build areas from groups
    areas = []
    for group in groups:
        group_els = [els[i] for i in group]
        rep = next((e for e in group_els if e.label.strip()), None)
        if rep is None:
            rep = next((e for e in group_els if e.leads_to), group_els[0])
        if rep.label:
            label = rep.label[:8]
        elif rep.icon_semantic and rep.icon_semantic in ICON_LABEL:
            label = ICON_LABEL[rep.icon_semantic]
        elif rep.leads_to:
            label = rep.leads_to[:6]
        else:
            label = "图标"
        areas.append(InteractiveArea(
            label=label,
            target_page=rep.leads_to or "",
            description=rep.leads_to or "",
            center_xy=[round(rep.x, 1), round(rep.y, 1)],
            element_type=rep.element_type,
        ))

    areas.sort(key=lambda a: a.center_xy[1])
    return areas


SYSTEM_PROMPT = """\
你是一个 iPhone 页面分析器。仔细分析截图，输出页面身份和所有可交互元素。

坐标系：左上角 (0, 0)，右下角 (1000, 1000)。坐标代表元素的视觉中心。

## ⚠️ 首要检查：是否是小程序 / H5 WebView
**在做任何其他分析之前，先看右上角**：
- 如果右上角有一个**胶囊形按钮**（pill / 圆角矩形轮廓，内含两个区域）：
  - 左区域：···（三个点，更多菜单）
  - 右区域：× 或 ✕（关闭符号）
- 这是微信小程序或其他 App 内嵌 H5 的专属胶囊按钮。
- 只要看到这个胶囊按钮，**立即将 is_miniprogram 设为 true**，无需满足其他条件。
- 同时，必须将胶囊右侧的 × 单独输出为一条 icon 元素（icon_semantic="close"，坐标指向 × 的中心）。
- 普通 iOS 原生页面没有这个胶囊按钮，is_miniprogram 填 false。

## 底部导航栏（bottom_nav）
判断页面是否有底部导航栏（tab bar / 底部操作栏）。有则 has_nav=true，无则 false。
底部导航栏内的元素（tab）点击后通常只切换页面，不需要返回。

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
        if not matched and el.element_type == "icon" and el.icon_semantic == "add":
            candidates = [
                (bi, bbox) for bi, bbox in enumerate(icon_bboxes)
                if bi not in matched_boxes
                and bbox.cx / img_w * 1000 >= 750
                and 120 <= bbox.cy / img_h * 1000 <= 220
            ]
            if candidates:
                bi, bbox = max(candidates, key=lambda item: item[1].conf)
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
        app_name=page.app_name,
        page_title=page.page_title,
        page_type=page.page_type,
        signature=page.signature,
        is_miniprogram=page.is_miniprogram,
        description=page.description,
        bottom_nav=page.bottom_nav,
        interactive_elements=merged,
    )


class PageParser:
    """Parse a screenshot into structured page identity and interactive elements."""

    _N_ATTEMPTS = 1

    def parse_screen(self, png_bytes: bytes) -> ParsedPage:
        """Full pipeline: LLM parse → YOLO detect → point-in-box merge."""
        import io

        from PIL import Image

        from policy_expr.recon.icon_detector import IconDetector

        page = self.parse(png_bytes)

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        w, h = img.size
        det = IconDetector(conf=0.1)
        boxes = det.detect_filtered(png_bytes, w, h)
        return enrich_with_icons(page, boxes, w, h)

    def analyze_screen(self, png_bytes: bytes) -> PageKnowledge:
        """Full pipeline: parse + YOLO merge + classify areas."""
        import io

        from PIL import Image

        from policy_expr.recon.icon_detector import IconDetector

        page = self.parse(png_bytes)

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        w, h = img.size
        det = IconDetector(conf=0.1)
        boxes = det.detect_filtered(png_bytes, w, h)

        merged = enrich_with_icons(page, boxes, w, h)
        # print(f"  LLM: {len(page.interactive_elements)} 个 | "
        #       f"YOLO: {len(boxes)} 个 | "
        #       f"融合: {len(merged.interactive_elements)} 个")

        areas = classify_elements(merged)
        # print(f"  区域数: {len(areas)}")

        return PageKnowledge(
            page=merged,
            areas=areas,
            llm_page=page,
            yolo_boxes=boxes,
            img_size=(w, h),
        )

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

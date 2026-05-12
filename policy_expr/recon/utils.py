"""Shared data classes, visualization, and output utilities for page recon."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from policy_expr.recon.page_parser import ParsedPage


# ── Data classes ──────────────────────────────────────────

@dataclass
class TapResult:
    """Result of tapping one element on the page."""
    index: int
    element_type: str
    label: str
    x: float
    y: float
    tap_ok: bool
    screenshot_path: str
    navigated: bool = False


@dataclass
class ReconResult:
    """Full recon result for one page."""
    app_name: str
    page_title: str
    page_type: str
    signature: str
    description: str
    elements_count: int
    initial_screenshot_path: str = ""
    taps: list[TapResult] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "app_name": self.app_name,
            "page_title": self.page_title,
            "page_type": self.page_type,
            "signature": self.signature,
            "description": self.description,
            "elements_count": self.elements_count,
            "initial_screenshot": self.initial_screenshot_path,
            "taps": [
                {
                    "index": t.index,
                    "element_type": t.element_type,
                    "label": t.label,
                    "x": t.x,
                    "y": t.y,
                    "tap_ok": t.tap_ok,
                    "navigated": t.navigated,
                    "screenshot": t.screenshot_path,
                }
                for t in self.taps
            ],
        }, ensure_ascii=False, indent=2))


# ── Visualization ─────────────────────────────────────────

TYPE_COLOR: dict[str, tuple[int, int, int]] = {
    "back_button": (255, 80, 80),
    "tab": (80, 160, 255),
    "button": (80, 220, 80),
    "link": (255, 200, 0),
    "input": (200, 80, 255),
    "menu_item": (255, 140, 0),
    "icon": (0, 220, 220),
}
RADIUS = 12
SCREEN_MATCH_SIZE = 64
SCREEN_MATCH_THRESHOLD = 0.99
SCREEN_DIFFERENT_THRESHOLD = 0.97


@dataclass(frozen=True)
class ScreenMatchDecision:
    """Layered decision for whether a screen matches the initial page."""
    matched: bool | None
    similarity: float
    method: str
    reason: str


def png_similarity(png1: bytes, png2: bytes, size: int = SCREEN_MATCH_SIZE) -> float:
    """Return grayscale pixel similarity between two PNG images."""
    img1 = Image.open(io.BytesIO(png1)).convert("L").resize((size, size))
    img2 = Image.open(io.BytesIO(png2)).convert("L").resize((size, size))
    total = sum(abs(int(a) - int(b)) for a, b in zip(img1.getdata(), img2.getdata()))
    return 1.0 - total / (255 * size * size)


def matches_initial(
    initial_png: bytes,
    current_png: bytes,
    threshold: float = SCREEN_MATCH_THRESHOLD,
) -> tuple[bool, float]:
    """Return whether current PNG is close enough to the initial screen."""
    similarity = png_similarity(initial_png, current_png)
    return similarity >= threshold, similarity


def decide_by_similarity(initial_png: bytes, current_png: bytes) -> ScreenMatchDecision:
    """Use only image similarity when the result is clear, otherwise defer."""
    similarity = png_similarity(initial_png, current_png)
    if similarity >= SCREEN_MATCH_THRESHOLD:
        return ScreenMatchDecision(True, similarity, "pixel", "similarity above match threshold")
    if similarity <= SCREEN_DIFFERENT_THRESHOLD:
        return ScreenMatchDecision(False, similarity, "pixel", "similarity below different threshold")
    return ScreenMatchDecision(None, similarity, "pixel", "similarity in uncertain band")


def same_page_by_structure(initial_page: ParsedPage, current_page: ParsedPage) -> tuple[bool, str]:
    """Compare parsed page structure as a model-backed fallback."""
    initial_ident = initial_page.identity
    current_ident = current_page.identity
    if initial_ident.signature and initial_ident.signature == current_ident.signature:
        return True, "same signature"

    same_identity = (
        initial_ident.app_name == current_ident.app_name
        and initial_ident.page_title == current_ident.page_title
        and initial_ident.page_type == current_ident.page_type
        and initial_page.bottom_nav.has_nav == current_page.bottom_nav.has_nav
    )
    if same_identity:
        return True, "same identity fields"

    initial_labels = {
        (el.element_type, el.label.strip())
        for el in initial_page.interactive_elements
        if el.label.strip()
    }
    current_labels = {
        (el.element_type, el.label.strip())
        for el in current_page.interactive_elements
        if el.label.strip()
    }
    if initial_labels and current_labels:
        overlap = len(initial_labels & current_labels)
        union = len(initial_labels | current_labels)
        score = overlap / union
        if score >= 0.7 and initial_ident.app_name == current_ident.app_name:
            return True, f"element overlap {score:.2f}"

    return False, (
        f"different page structure: initial={initial_ident.signature!r}, "
        f"current={current_ident.signature!r}"
    )


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", size)
    except Exception:
        return ImageFont.load_default()


def visualize(page: ParsedPage, png_bytes: bytes) -> bytes:
    """Draw element markers on screenshot, return annotated PNG bytes."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font = _font(14)

    for i, el in enumerate(page.interactive_elements, 1):
        cx = int(el.x / 1000 * w)
        cy = int(el.y / 1000 * h)
        is_yolo_extra = (
            el.label == "" and el.element_type == "icon" and el.leads_to == ""
        )
        if is_yolo_extra:
            color = (180, 180, 180)
            draw.polygon(
                [(cx, cy - RADIUS), (cx + RADIUS, cy), (cx, cy + RADIUS), (cx - RADIUS, cy)],
                fill=(*color, 160),
                outline=(255, 255, 255, 255),
            )
            label = f"Y{i}"
        else:
            color = TYPE_COLOR.get(el.element_type, (200, 200, 200))
            draw.ellipse(
                [cx - RADIUS, cy - RADIUS, cx + RADIUS, cy + RADIUS],
                fill=(*color, 200),
                outline=(255, 255, 255, 255),
                width=2,
            )
            label = f"{i} {el.label[:12] if el.label else el.element_type}"
        draw.text((cx + RADIUS + 3, cy - 8), label, fill=(*color, 255), font=font)

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def print_result(page: ParsedPage) -> None:
    """Print parsed page to stdout."""
    ident = page.identity
    print(f"  应用   : {ident.app_name}")
    print(f"  标题   : {ident.page_title}")
    print(f"  类型   : {ident.page_type}")
    print(f"  指纹   : {ident.signature}")
    print(f"  描述   : {page.description}")
    print(f"  元素数 : {len(page.interactive_elements)}")
    for i, el in enumerate(page.interactive_elements, 1):
        icon_tag = f"  [{el.icon_semantic}]" if el.icon_semantic else ""
        print(
            f"    [{i:2d}] ({el.x:5.0f},{el.y:5.0f})  {el.element_type:<12}"
            f"{icon_tag:<14}  「{el.label}」→ {el.leads_to}"
        )


def viz_result(page: ParsedPage, png_bytes: bytes, stem: str, out_dir: Path) -> None:
    """Print and save parsed page result + visualization."""
    print_result(page)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}_result.json"
    json_path.write_text(page.model_dump_json(indent=2), encoding="utf-8")

    viz_path = out_dir / f"{stem}_viz.png"
    viz_path.write_bytes(visualize(page, png_bytes))

    print(f"\n  JSON : {json_path}")
    print(f"  可视化 : {viz_path}")

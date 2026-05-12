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
    taps: list[TapResult] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "app_name": self.app_name,
            "page_title": self.page_title,
            "page_type": self.page_type,
            "signature": self.signature,
            "description": self.description,
            "elements_count": self.elements_count,
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

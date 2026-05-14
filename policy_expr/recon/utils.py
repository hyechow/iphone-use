"""Shared data classes, visualization, and output utilities for page recon."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from policy_expr.recon.page_parser import ParsedPage


# ── Exceptions ────────────────────────────────────────────

class ProbeAbortedError(RuntimeError):
    """Raised when probe_elements cannot return to initial page after a tap."""
    def __init__(
        self,
        message: str,
        failed_tap: int,
        failed_element: str,
        back_attempts: list[dict],
    ):
        super().__init__(message)
        self.failed_tap = failed_tap
        self.failed_element = failed_element
        self.back_attempts = back_attempts  # list of {strategy, coords, score, success}


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
    back_attempts: list[dict] = field(default_factory=list)


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
                    "back_attempts": t.back_attempts,
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
    """Return edge IoU between two PNG images (robust to dynamic content changes)."""
    from skimage.feature import canny

    img1 = np.array(Image.open(io.BytesIO(png1)).convert("L"), dtype=np.float64) / 255.0
    img2_raw = Image.open(io.BytesIO(png2)).convert("L")
    if img1.shape != img2_raw.size[::-1]:
        img2_raw = img2_raw.resize((img1.shape[1], img1.shape[0]))
    img2 = np.array(img2_raw, dtype=np.float64) / 255.0
    e1 = canny(img1).astype(np.float64)
    e2 = canny(img2).astype(np.float64)
    intersection = (e1 * e2).sum()
    union = (e1 + e2).clip(0, 1).sum()
    return float(intersection / union) if union > 0 else 0.0


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


GROUP_PALETTE = [
    (231, 76, 60),    # red
    (46, 204, 113),   # green
    (52, 152, 219),   # blue
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
    (230, 126, 34),   # orange
    (26, 188, 156),   # teal
    (236, 100, 165),  # pink
    (52, 73, 94),     # dark blue
    (127, 140, 141),  # gray
    (192, 57, 43),    # dark red
    (39, 174, 96),    # dark green
    (41, 128, 185),   # dark blue
    (243, 156, 18),   # dark yellow
    (142, 68, 173),   # dark purple
]


def visualize(page: ParsedPage, png_bytes: bytes) -> bytes:
    """Draw element markers on screenshot, return annotated PNG bytes."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    for i, el in enumerate(page.interactive_elements, 1):
        cx = int(el.x / 1000 * w)
        cy = int(el.y / 1000 * h)
        is_yolo_extra = (
            el.label == "" and el.element_type == "icon" and el.leads_to == ""
        )
        if is_yolo_extra:
            draw.polygon(
                [(cx, cy - RADIUS), (cx + RADIUS, cy), (cx, cy + RADIUS), (cx - RADIUS, cy)],
                fill=(180, 180, 180, 160),
                outline=(255, 255, 255, 255),
            )
        else:
            color = TYPE_COLOR.get(el.element_type, (200, 200, 200))
            draw.ellipse(
                [cx - RADIUS, cy - RADIUS, cx + RADIUS, cy + RADIUS],
                fill=(*color, 200),
                outline=(255, 255, 255, 255),
                width=2,
            )

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def visualize_yolo(
    png_bytes: bytes,
    boxes: list,
    img_w: int,
    img_h: int,
) -> bytes:
    """Draw YOLO detected icon bboxes on screenshot."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    for b in boxes:
        draw.rectangle(
            [int(b.x1), int(b.y1), int(b.x2), int(b.y2)],
            outline=(0, 220, 220, 255),
            width=2,
        )

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def visualize_areas(
    png_bytes: bytes,
    areas: list,
) -> bytes:
    """Draw area markers on screenshot."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    for ai, area in enumerate(areas):
        color = GROUP_PALETTE[ai % len(GROUP_PALETTE)]
        cx = int(area.center_xy[0] / 1000 * w)
        cy = int(area.center_xy[1] / 1000 * h)
        draw.ellipse(
            [cx - RADIUS, cy - RADIUS, cx + RADIUS, cy + RADIUS],
            fill=(*color, 200),
            outline=(255, 255, 255, 255),
            width=2,
        )

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def print_areas(knowledge: "PageKnowledge") -> None:  # noqa: F821
    """Print areas to stdout."""
    ident = knowledge.page.identity
    # print(f"  应用 : {ident.app_name}")
    # print(f"  页面 : {ident.page_title}")
    # print(f"  指纹 : {ident.signature}")
    # print(f"  区域数 : {len(knowledge.areas)}")
    # for i, a in enumerate(knowledge.areas, 1):
    #     print(f"    [{i:2d}] ({a.center_xy[0]:5.0f},{a.center_xy[1]:5.0f})  "
    #           f"「{a.label}」→ {a.target_page}")


def viz_result(
    knowledge: "PageKnowledge",  # noqa: F821
    png_bytes: bytes,
    stem: str,
    out_dir: Path,
) -> None:
    """Save area JSON + LLM/YOLO/area visualizations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print_areas(knowledge)

    output = knowledge.model_dump(mode="json")
    output["page"].pop("interactive_elements", None)

    json_path = out_dir / f"{stem}_result.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    # LLM-only visualization
    if knowledge.llm_page:
        llm_viz_path = out_dir / f"{stem}_llm_viz.png"
        llm_viz_path.write_bytes(visualize(knowledge.llm_page, png_bytes))
        # print(f"  LLM 可视化 : {llm_viz_path}")

    # YOLO-only visualization
    if knowledge.yolo_boxes and knowledge.img_size:
        yolo_viz_path = out_dir / f"{stem}_yolo_viz.png"
        yolo_viz_path.write_bytes(visualize_yolo(
            png_bytes, knowledge.yolo_boxes,
            knowledge.img_size[0], knowledge.img_size[1],
        ))
        # print(f"  YOLO 可视化 : {yolo_viz_path}")

    # Area visualization
    areas_viz_path = out_dir / f"{stem}_areas_viz.png"
    areas_viz_path.write_bytes(visualize_areas(png_bytes, knowledge.areas))

    # print(f"  JSON : {json_path}")
    # print(f"  区域可视化 : {areas_viz_path}")

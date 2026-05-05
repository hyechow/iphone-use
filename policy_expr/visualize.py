"""Policy decision reporting and visualization."""

import io
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from policy_expr.schemas import Action, ActionDecision

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "logs" / "policy_expr" / "single-step" / "structured_output_result.png"


def print_decision(
    decision: ActionDecision,
    png_bytes: bytes,
    output_path: Path = OUTPUT,
) -> None:
    action = decision.action
    coords = f"  ({action.x:.0f},{action.y:.0f})" if action.x is not None else ""
    text = f"  文字={action.text!r}" if action.text else ""
    print(f"\n屏幕  : {decision.summary}")
    print(f"\n推理  : {decision.reasoning}")
    print("\n" + "=" * 50)
    print(f"[{action.action_type}] {action.description}{coords}{text}")
    print("=" * 50)
    visualize(png_bytes, action, output_path)


def visualize(png_bytes: bytes, action: Action, output_path: Path = OUTPUT) -> None:
    if action.x is None or action.y is None:
        return

    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 28)
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 20)
    except Exception:
        title_font = ImageFont.load_default()
        font = ImageFont.load_default()

    px = action.x / 1000 * img.width
    py = action.y / 1000 * img.height
    radius = max(34, int(min(img.width, img.height) * 0.045))
    red = (255, 40, 40, 255)
    yellow = (255, 224, 64, 255)
    shadow = (0, 0, 0, 210)

    draw.ellipse(
        [px - radius - 14, py - radius - 14, px + radius + 14, py + radius + 14],
        fill=(255, 40, 40, 80),
    )
    draw.ellipse(
        [px - radius, py - radius, px + radius, py + radius],
        outline=yellow,
        width=8,
    )
    draw.ellipse(
        [px - radius + 8, py - radius + 8, px + radius - 8, py + radius - 8],
        outline=red,
        width=5,
    )
    draw.line(
        [px - radius - 22, py, px + radius + 22, py],
        fill=red,
        width=5,
    )
    draw.line(
        [px, py - radius - 22, px, py + radius + 22],
        fill=red,
        width=5,
    )
    draw.ellipse(
        [px - 8, py - 8, px + 8, py + 8],
        fill=yellow,
    )

    label_lines = [
        f"{action.action_type.upper()} ({action.x:.0f}, {action.y:.0f})",
        *textwrap.wrap(action.description, width=18),
    ]
    line_heights = [title_font.getbbox(label_lines[0])[3] + 8]
    line_heights.extend(font.getbbox(line)[3] + 8 for line in label_lines[1:])
    label_w = min(
        max(
            title_font.getbbox(label_lines[0])[2],
            *(font.getbbox(line)[2] for line in label_lines[1:] or [""]),
        )
        + 28,
        img.width - 24,
    )
    label_h = sum(line_heights) + 22
    label_x = px + radius + 28
    if label_x + label_w > img.width - 12:
        label_x = px - radius - 28 - label_w
    label_x = max(12, min(label_x, img.width - label_w - 12))
    label_y = max(12, min(py - label_h / 2, img.height - label_h - 12))

    target_edge_x = px + radius if label_x > px else px - radius
    label_edge_x = label_x if label_x > px else label_x + label_w
    label_mid_y = label_y + label_h / 2
    draw.line([target_edge_x, py, label_edge_x, label_mid_y], fill=yellow, width=5)
    draw.rounded_rectangle(
        [label_x + 4, label_y + 4, label_x + label_w + 4, label_y + label_h + 4],
        radius=12,
        fill=(0, 0, 0, 110),
    )
    draw.rounded_rectangle(
        [label_x, label_y, label_x + label_w, label_y + label_h],
        radius=12,
        fill=shadow,
        outline=yellow,
        width=3,
    )
    text_y = label_y + 14
    draw.text((label_x + 14, text_y), label_lines[0], fill=yellow, font=title_font)
    text_y += line_heights[0]
    for line in label_lines[1:]:
        draw.text((label_x + 14, text_y), line, fill=(255, 255, 255, 255), font=font)
        text_y += font.getbbox(line)[3] + 8

    result = Image.alpha_composite(img, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"可视化已保存: {output_path}")

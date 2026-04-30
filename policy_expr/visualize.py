"""Policy decision reporting and visualization."""

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from policy_expr.schemas import Action, PolicyDecision

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "images" / "structured_output_result.png"


def print_decision(decision: PolicyDecision, png_bytes: bytes) -> None:
    action = decision.action
    coords = f"  ({action.x:.0f},{action.y:.0f})" if action.x is not None else ""
    text = f"  文字={action.text!r}" if action.text else ""
    print(f"\n屏幕  : {decision.summary}")
    print(f"\n推理  : {decision.reasoning}")
    print("\n" + "=" * 50)
    print(f"[{action.action_type}] {action.description}{coords}{text}")
    print("=" * 50)
    visualize(png_bytes, action)


def visualize(png_bytes: bytes, action: Action, output_path: Path = OUTPUT) -> None:
    if action.x is None or action.y is None:
        return

    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 22)
    except Exception:
        font = ImageFont.load_default()

    px = action.x / 1000 * img.width
    py = action.y / 1000 * img.height
    radius = 30
    draw.ellipse(
        [px - radius, py - radius, px + radius, py + radius],
        fill=(255, 50, 50, 120),
    )
    draw.ellipse(
        [px - radius, py - radius, px + radius, py + radius],
        outline=(255, 50, 50, 255),
        width=3,
    )
    draw.line(
        [px - radius - 8, py, px + radius + 8, py],
        fill=(255, 50, 50, 255),
        width=3,
    )
    draw.line(
        [px, py - radius - 8, px, py + radius + 8],
        fill=(255, 50, 50, 255),
        width=3,
    )
    draw.text(
        (px + radius + 8, py - 14),
        action.description,
        fill=(255, 50, 50, 255),
        font=font,
    )

    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(output_path)
    print(f"可视化已保存: {output_path}")


"""Save per-action visualizations for ReAct loops."""
import base64
import io
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from agent.tool_args import coerce_number, normalize_tool_args


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


class ReActVisualizer:
    """Annotates the latest screenshot for each selected tool action."""

    def __init__(self, thread_id: str, turn_id: str, root: str | Path = "logs"):
        self.turn_id = _safe_path_part(turn_id)
        self.thread_id = _safe_path_part(thread_id)
        self.output_dir = Path(root) / self.thread_id / self.turn_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._latest_screenshot_b64: str | None = None
        self._index = 0

    def update_screenshot(self, screenshot_b64: str) -> None:
        self._latest_screenshot_b64 = screenshot_b64

    def save_action(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        screenshot_b64: str | None = None,
    ) -> Path | None:
        screenshot_b64 = screenshot_b64 or self._latest_screenshot_b64
        if not screenshot_b64:
            return None

        args = normalize_tool_args(name, args)
        self._index += 1
        img = self._load_image(screenshot_b64)
        draw = ImageDraw.Draw(img)
        font = self._font(16)
        small_font = self._font(13)

        title = f"{self._index}. {name}"
        detail = self._format_args(args)
        draw.rectangle((8, 8, img.width - 8, 58), fill=(0, 0, 0, 170))
        draw.text((18, 14), title, fill=(255, 255, 255, 255), font=font)
        if detail:
            draw.text((18, 36), detail, fill=(230, 230, 230, 255), font=small_font)

        if name == "tap_screen":
            self._draw_tap(draw, img, args)
        elif name == "go_to_home_screen":
            self._draw_home(draw, img)
        elif name == "type_text":
            self._draw_text_action(draw, img, args)

        output_path = self.output_dir / f"{self._index}.jpg"
        img.convert("RGB").save(output_path, format="JPEG", quality=90)
        return output_path

    @staticmethod
    def _load_image(screenshot_b64: str) -> Image.Image:
        raw = base64.b64decode(screenshot_b64)
        return Image.open(io.BytesIO(raw)).convert("RGBA")

    @staticmethod
    def _font(size: int):
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except OSError:
            return ImageFont.load_default()

    @staticmethod
    def _format_args(args: dict[str, Any]) -> str:
        if not args:
            return ""
        parts = []
        for key, value in args.items():
            if isinstance(value, str) and len(value) > 40:
                value = value[:37] + "..."
            parts.append(f"{key}={value}")
        return ", ".join(parts)

    @staticmethod
    def _draw_tap(draw: ImageDraw.ImageDraw, img: Image.Image, args: dict[str, Any]) -> None:
        x = coerce_number(args.get("x")) / 1000 * img.width
        y = coerce_number(args.get("y")) / 1000 * img.height
        r = 26
        color = (234, 67, 53, 255)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=4)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        draw.line((x - r - 10, y, x + r + 10, y), fill=color, width=2)
        draw.line((x, y - r - 10, x, y + r + 10), fill=color, width=2)

    @staticmethod
    def _draw_home(draw: ImageDraw.ImageDraw, img: Image.Image) -> None:
        x, y = img.width / 2, img.height - 24
        color = (52, 168, 83, 255)
        draw.rounded_rectangle((x - 70, y - 10, x + 70, y + 10), radius=10, outline=color, width=4)

    def _draw_text_action(self, draw: ImageDraw.ImageDraw, img: Image.Image, args: dict[str, Any]) -> None:
        text = str(args.get("text", ""))
        if len(text) > 80:
            text = text[:77] + "..."
        font = self._font(14)
        y = img.height - 58
        draw.rectangle((8, y, img.width - 8, img.height - 8), fill=(240, 165, 0, 210))
        draw.text((18, y + 16), f"type: {text}", fill=(0, 0, 0, 255), font=font)

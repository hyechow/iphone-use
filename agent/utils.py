"""Agent utility functions."""
import base64
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ocrmac.ocrmac import OCR

# Default icon detection model path (exported by ultralytics)
_DEFAULT_ICON_MODEL = Path.home() / ".mirroir-mcp/models/icon_detect_download/icon_detect/model.mlpackage"
_icon_model = None

LANGUAGE_PREFERENCE = ["zh-Hans", "en-US"]


@dataclass
class OcrResult:
    text: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    def tap_coords(self, width: int, height: int, y_offset: float = 0.0) -> tuple[float, float]:
        """Convert to tap tool coords (top-left origin, logical pixels).
        Pass window size, not screenshot size (Retina screenshots are 2×).
        y_offset: normalized offset applied before conversion, negative = upward."""
        px = self.center_x * width
        py = (1.0 - (self.center_y + y_offset)) * height
        return px, py


def ocr_from_bytes(png_bytes: bytes) -> tuple[list[OcrResult], tuple[int, int]]:
    """Run OCR on raw PNG bytes. Returns (results, (width, height))."""
    import io
    from PIL import Image

    image = Image.open(io.BytesIO(png_bytes))
    results = OCR(image, language_preference=LANGUAGE_PREFERENCE).recognize()
    ocr_results = [
        OcrResult(text=text, confidence=conf, x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
        for text, conf, bbox in results
    ]
    return ocr_results, image.size


def ocr_from_b64(b64: str) -> tuple[list[OcrResult], tuple[int, int]]:
    """Run OCR on a base64-encoded PNG string. Returns (results, (width, height))."""
    return ocr_from_bytes(base64.b64decode(b64))


def home_indicator_coords(win_w: int, win_h: int) -> tuple[float, float]:
    """Return tap coords for the iPhone home indicator bar (bottom center).
    Tap this to return to the home screen. 16px from bottom edge."""
    return win_w / 2, win_h - 16.0


def paste_text(text: str) -> None:
    """Write text to macOS clipboard and send Cmd+V (supports Chinese and all Unicode)."""
    subprocess.run(["pbcopy"], input=text.encode(), check=True)
    subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to keystroke "v" using command down'
    ], check=True)


def is_home_screen(results: list[OcrResult]) -> bool:
    """Heuristic: detect if current screen is the iPhone home screen.

    Rules:
    1. Has a clock string (HH:MM) near the top (Vision y > 0.85)
    2. Has a search bar ("搜索") near the bottom (Vision y < 0.2)
    """
    has_clock = any(
        re.fullmatch(r"\d{1,2}:\d{2}", r.text) and r.y > 0.85
        for r in results
    )
    has_search = any("搜索" in r.text and r.y < 0.2 for r in results)

    return has_clock and has_search


# ---------------------------------------------------------------------------
# Icon detection (YOLO via CoreML)
# ---------------------------------------------------------------------------

@dataclass
class IconResult:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def _get_icon_model(model_path: str | Path | None = None):
    """Load (and cache) the icon detection model."""
    global _icon_model
    if _icon_model is None:
        from ultralytics import YOLO
        path = model_path or _DEFAULT_ICON_MODEL
        if not Path(path).exists():
            raise FileNotFoundError(f"Icon model not found: {path}")
        _icon_model = YOLO(str(path))
    return _icon_model


def detect_icons(
    image_or_path: str | Path | bytes,
    conf: float = 0.3,
    model_path: str | Path | None = None,
) -> list[IconResult]:
    """Detect icons in an image using the YOLO CoreML model.

    Args:
        image_or_path: file path (str/Path) or raw PNG bytes.
        conf: minimum confidence threshold (default 0.3).
        model_path: override default model path.

    Returns:
        List of IconResult sorted by confidence descending.
    """
    model = _get_icon_model(model_path)
    results = model.predict(source=image_or_path, conf=conf, verbose=False)
    icons = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        icons.append(IconResult(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(box.conf[0])))
    icons.sort(key=lambda i: i.confidence, reverse=True)
    return icons

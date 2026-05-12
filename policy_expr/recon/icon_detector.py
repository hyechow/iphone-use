"""YOLO-based icon detector for UI screenshots."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

MODEL_PATH = Path("models/omniparser_v2/icon_detect/model.pt")
DEFAULT_CONF = 0.3


@dataclass
class IconBbox:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


class IconDetector:
    """Wraps the OmniParser YOLO icon detection model."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        conf: float = DEFAULT_CONF,
    ) -> None:
        from ultralytics import YOLO  # deferred import: YOLO cold-start is slow
        self._model = YOLO(str(model_path))
        self._conf = conf

    def detect(self, png_bytes: bytes) -> list[IconBbox]:
        """Return icon bounding boxes in pixel coordinates of the input image."""
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        results = self._model(img, conf=self._conf, verbose=False)
        bboxes: list[IconBbox] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bboxes.append(IconBbox(x1, y1, x2, y2, float(box.conf[0])))
        return bboxes

    def detect_filtered(
        self,
        png_bytes: bytes,
        img_w: int,
        img_h: int,
        nav_bar_y: int = 200,
        max_width_ratio: float = 0.8,
        overlap_thresh: float = 0.8,
        min_gray_std: float = 5.0,
    ) -> list[IconBbox]:
        """Detect icons and apply three-layer filtering.

        1. Remove navigation bar noise (cy < nav_bar_y)
        2. Remove wide boxes (width > max_width_ratio * img_w)
        3. Remove visually blank boxes (low grayscale standard deviation)
        4. Merge small boxes whose area >= overlap_thresh is inside a larger box
        """
        boxes = self.detect(png_bytes)

        # Filter 1: navigation bar
        boxes = [b for b in boxes if b.cy >= nav_bar_y]

        # Filter 2: overly wide
        max_w = img_w * max_width_ratio
        boxes = [b for b in boxes if (b.x2 - b.x1) <= max_w]

        # Filter 3: visually blank boxes
        gray = Image.open(io.BytesIO(png_bytes)).convert("L")
        boxes = [
            b for b in boxes
            if _gray_std(gray, b) >= min_gray_std
        ]

        # Filter 4: overlap-based dedup (larger boxes absorb smaller ones)
        def _overlap_ratio(small: IconBbox, big: IconBbox) -> float:
            ix1 = max(small.x1, big.x1)
            iy1 = max(small.y1, big.y1)
            ix2 = min(small.x2, big.x2)
            iy2 = min(small.y2, big.y2)
            if ix1 >= ix2 or iy1 >= iy2:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            area = (small.x2 - small.x1) * (small.y2 - small.y1)
            return inter / area if area > 0 else 0.0

        sorted_boxes = sorted(boxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1), reverse=True)
        covered: set[int] = set()
        for i, big in enumerate(sorted_boxes):
            for j in range(i + 1, len(sorted_boxes)):
                if j in covered:
                    continue
                if _overlap_ratio(sorted_boxes[j], big) >= overlap_thresh:
                    covered.add(j)
        return [b for i, b in enumerate(sorted_boxes) if i not in covered]


def _gray_std(img: Image.Image, box: IconBbox) -> float:
    crop = img.crop((
        max(0, int(box.x1)),
        max(0, int(box.y1)),
        min(img.width, int(box.x2)),
        min(img.height, int(box.y2)),
    ))
    if crop.width <= 0 or crop.height <= 0:
        return 0.0
    hist = crop.histogram()
    total = sum(hist)
    if total <= 0:
        return 0.0
    mean = sum(i * count for i, count in enumerate(hist)) / total
    variance = sum(((i - mean) ** 2) * count for i, count in enumerate(hist)) / total
    return variance ** 0.5

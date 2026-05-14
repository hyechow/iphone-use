"""YOLO calibrator: detect icons once, find nearest to any target point."""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from policy_expr.recon.icon_detector import IconBbox, IconDetector


@dataclass
class YoloCalibrator:
    """Run YOLO detection once, then query nearest icon to any target.

    All coordinates are in normalized 0-1000 space.
    """
    boxes: list[IconBbox]
    img_w: int
    img_h: int

    @classmethod
    def from_png(cls, png_bytes: bytes, conf: float = 0.1) -> YoloCalibrator | None:
        """Create calibrator from screenshot. Returns None if no icons detected."""
        det = IconDetector(conf=conf)
        boxes = det.detect(png_bytes)
        if not boxes:
            return None
        img = Image.open(io.BytesIO(png_bytes))
        return cls(boxes, img.width, img.height)

    def nearest(self, target_x: float, target_y: float, max_dist: float = 80.0) -> tuple[float, float] | None:
        """Find the detected icon nearest to (target_x, target_y) within max_dist."""
        best, best_dist = None, float("inf")
        for b in self.boxes:
            nx = b.cx / self.img_w * 1000
            ny = b.cy / self.img_h * 1000
            dist = ((nx - target_x) ** 2 + (ny - target_y) ** 2) ** 0.5
            if dist >= best_dist or dist > max_dist:
                continue
            best_dist = dist
            best = (nx, ny)
        return best

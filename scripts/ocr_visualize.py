"""可视化 OCR 识别结果和 tap 点击位置"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
from agent.utils import ocr_from_bytes

SCREENSHOT = Path(__file__).parent.parent / "screenshot.png"
OUTPUT = Path(__file__).parent.parent / "ocr_vis.png"
TARGET = "美团"  # 高亮显示的目标文字


def draw(png_bytes: bytes, target_text: str, out_path: Path):
    ocr_results, (img_w, img_h) = ocr_from_bytes(png_bytes)
    win_w, win_h = img_w // 2, img_h // 2

    img = Image.open(__import__("io").BytesIO(png_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 22)
        font_small = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 18)
    except Exception:
        font = font_small = ImageFont.load_default()

    for r in ocr_results:
        is_target = target_text in r.text
        # bbox 转像素（图片原始 2× 分辨率）
        x1 = r.x * img_w
        y1 = (1.0 - r.y - r.height) * img_h
        x2 = (r.x + r.width) * img_w
        y2 = (1.0 - r.y) * img_h

        color = (255, 80, 80, 180) if is_target else (80, 200, 80, 120)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, y1 + 2), r.text, fill=color, font=font_small)

        if is_target:
            # tap 点（含 y_offset=0.06）
            tx, ty = r.tap_coords(win_w, win_h, y_offset=0.06)
            # 转回图片坐标（2×）
            px, py = tx * 2, ty * 2
            R = 20
            draw.ellipse([px - R, py - R, px + R, py + R], fill=(255, 0, 0, 200))
            draw.line([px - R, py, px + R, py], fill="white", width=3)
            draw.line([px, py - R, px, py + R], fill="white", width=3)
            draw.text((px + R + 4, py - 12), f"tap ({tx:.0f},{ty:.0f})", fill=(255, 0, 0, 255), font=font)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(out_path)
    print(f"已保存: {out_path}")


if not SCREENSHOT.exists():
    print("screenshot.png 不存在，请先运行 scripts/screenshot_test.py")
    sys.exit(1)

draw(SCREENSHOT.read_bytes(), TARGET, OUTPUT)

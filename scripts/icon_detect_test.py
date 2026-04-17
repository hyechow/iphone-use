"""测试图标检测功能（YOLO CoreML）"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
from agent.utils import detect_icons

SCREENSHOT = Path(__file__).parent.parent / "images" / "screenshot.png"
OUTPUT = Path(__file__).parent.parent / "images" / "icon_detect_vis.png"

if not SCREENSHOT.exists():
    print("screenshot.png 不存在，请先运行 scripts/screenshot_test.py")
    sys.exit(1)

print(f"图片路径: {SCREENSHOT}")
print("-" * 50)

# 预热
detect_icons(str(SCREENSHOT))

# 推理计时
start = time.perf_counter()
icons = detect_icons(str(SCREENSHOT), conf=0.3)
elapsed = (time.perf_counter() - start) * 1000

print(f"推理耗时: {elapsed:.1f} ms")
print(f"检测到 {len(icons)} 个图标:\n")
for i, icon in enumerate(icons):
    cx, cy = icon.center
    print(f"  {i+1:>2}. conf={icon.confidence:.2f}  center=({cx:.0f}, {cy:.0f})  "
          f"bbox=[{icon.x1:.0f}, {icon.y1:.0f}, {icon.x2:.0f}, {icon.y2:.0f}]")

# 可视化
img = Image.open(SCREENSHOT).convert("RGBA")
overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

try:
    font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 20)
except Exception:
    font = ImageFont.load_default()

for icon in icons:
    color = (0, 150, 255, 180) if icon.confidence >= 0.5 else (255, 165, 0, 140)
    draw.rectangle(
        [icon.x1, icon.y1, icon.x2, icon.y2],
        outline=color, width=3,
    )
    label = f"{icon.confidence:.2f}"
    draw.text((icon.x1 + 3, icon.y1 + 2), label, fill=color, font=font)

result = Image.alpha_composite(img, overlay).convert("RGB")
result.save(OUTPUT)
print(f"\n可视化已保存: {OUTPUT}")

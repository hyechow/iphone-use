"""测试 ocrmac（Apple Vision OCR）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocrmac.ocrmac import OCR

screenshot_path = Path(__file__).parent.parent / "images" / "screenshot.png"

if not screenshot_path.exists():
    print("screenshot.png 不存在，请先运行 scripts/screenshot_test.py")
    sys.exit(1)

print(f"图片路径: {screenshot_path}")
print("-" * 40)

results = OCR(str(screenshot_path)).recognize()

if not results:
    print("未识别到文字")
else:
    print(f"识别到 {len(results)} 个文字元素：\n")
    for text, confidence, bbox in results:
        print(f"  [{confidence:.2f}] {text!r}  bbox={bbox}")

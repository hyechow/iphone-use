"""验证通用滚动接口 scroll_screen。

脚本会保存每次调用前后的截图，并输出 scroll_screen 内部的
swipe / drag fallback 诊断，重点看 mean_diff 是否足够大。
"""

import sys
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.sessions import get_client
from agent.tools import scroll_screen

OUT_DIR = Path(__file__).parent.parent / "images"
THREAD_ID = "scroll-test"


def screenshot(name: str) -> Path:
    client = get_client(THREAD_ID)
    png = client.screenshot()
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"scroll_{name}.png"
    path.write_bytes(png)
    print(f"  截图: {path.name}")
    return path


def image_diff(before: Path, after: Path) -> float:
    before_img = Image.open(before).convert("RGB")
    after_img = Image.open(after).convert("RGB")
    diff = ImageChops.difference(before_img, after_img)
    return sum(ImageStat.Stat(diff).mean) / 3


def run_scroll_case(
    label: str,
    direction: str,
    max_attempts: int = 4,
) -> Path:
    before = screenshot(f"{label}_before")
    print(f"\n=== {label}: direction={direction} ===")
    result = scroll_screen.invoke(
        {
            "direction": direction,
            "max_attempts": max_attempts,
        },
        config={"configurable": {"thread_id": THREAD_ID}},
    )
    print(result)
    time.sleep(0.8)
    after = screenshot(f"{label}_after")
    print(f"  总图像变化: mean_diff={image_diff(before, after):.2f}")
    return after


def main() -> None:
    client = get_client(THREAD_ID)
    print(client.status())

    print("\n=== 初始截图 ===")
    screenshot("00_initial")

    run_scroll_case("01_up", "up", max_attempts=4)
    run_scroll_case("02_down", "down", max_attempts=4)
    run_scroll_case("03_up_again", "up", max_attempts=4)

    print("\n完成。查看 images/scroll_*.png，并重点看每段输出中的 mean_diff。")


if __name__ == "__main__":
    main()

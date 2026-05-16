"""Mini-program detection + close test.

Usage:
    uv run python scripts/test_miniprogram.py

Flow:
    1. Connect to phone (user must already be on a mini-program page).
    2. Take screenshot, detect capsule via GUIClip (no LLM needed).
    3. If detected, parse page to find × coordinates and tap to close.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from policy_expr.perception import LivePhoneSession
from policy_expr.executor import logical_xy
from policy_expr.recon.page_parser import PageParser, detect_miniprogram


def main() -> None:
    parser = PageParser()

    with LivePhoneSession() as phone:
        assert phone.client is not None

        print("\n" + "=" * 60)
        print("  截取当前页面...")
        print("=" * 60)
        png = phone.screenshot()

        # Step 1: GUIClip capsule detection (instant, no LLM)
        is_mini = detect_miniprogram(png)
        print(f"\n  is_miniprogram: {is_mini}")

        if not is_mini:
            print("  未检测到小程序，退出。")
            return

        # Step 2: Parse page to get × button coordinates (needs LLM)
        print("  解析页面元素（获取胶囊 × 坐标）...")
        page = parser.parse_screen(png)
        print(f"  description: {page.description}")

        print(f"  interactive_elements ({len(page.interactive_elements)} 个):")
        for el in page.interactive_elements:
            sem = f"  [{el.icon_semantic}]" if el.icon_semantic else ""
            print(f"    ({el.x:>5.0f},{el.y:>4.0f})  {el.element_type:12s}{sem}  {el.label or '(无标签)'}")

        # Find rightmost icon in top-right area — capsule × is always the rightmost
        top_right_icons = [
            e for e in page.interactive_elements
            if e.x > 800 and e.y < 200
        ]
        if top_right_icons:
            close_el = max(top_right_icons, key=lambda e: e.x)
            ax, ay = close_el.x, close_el.y
            ax = min(ax + 25, 970)
        else:
            ax, ay = 950.0, 130.0
        lx, ly = logical_xy(ax, ay)
        print(f"  逻辑坐标: ({lx:.0f}, {ly:.0f})")

        input("\n  按回车执行关闭...")
        phone.client.tap(lx, ly)
        time.sleep(1.5)

        after = phone.screenshot()
        after_path = Path("/tmp/miniprogram_after_close.png")
        after_path.write_bytes(after)
        print(f"  关闭后截图: {after_path}")
        subprocess.Popen(["open", str(after_path)])
        print("  完成。")


if __name__ == "__main__":
    main()

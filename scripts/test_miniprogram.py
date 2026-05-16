"""Mini-program detection + close test.

Usage:
    uv run python scripts/test_miniprogram.py

Flow:
    1. Connect to phone (user must already be on a mini-program page).
    2. Parse the current screenshot with PageParser.
    3. Print identity fields including is_miniprogram.
    4. If mini-program detected, tap the × capsule button to close it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from policy_expr.perception import LivePhoneSession
from policy_expr.executor import logical_xy
from policy_expr.recon.page_parser import PageParser


def main() -> None:
    parser = PageParser()

    with LivePhoneSession() as phone:
        assert phone.client is not None

        print("\n" + "=" * 60)
        print("  截取当前页面...")
        print("=" * 60)
        png = phone.screenshot()

        print("  解析页面（LLM 调用中）...")
        page = parser.parse_screen(png)

        print(f"\n  app_name      : {page.app_name}")
        print(f"  page_title    : {page.page_title}")
        print(f"  page_type     : {page.page_type}")
        print(f"  is_miniprogram: {page.is_miniprogram}")
        print(f"  signature     : {page.signature}")

        print(f"\n  interactive_elements ({len(page.interactive_elements)} 个):")
        for el in page.interactive_elements:
            sem = f"  [{el.icon_semantic}]" if el.icon_semantic else ""
            print(f"    ({el.x:>5.0f},{el.y:>4.0f})  {el.element_type:12s}{sem}  {el.label or '(无标签)'}")

        if not page.is_miniprogram:
            print("\n  未检测到小程序，退出。")
            return

        print("\n" + "=" * 60)
        print("  检测到小程序，寻找胶囊 × 按钮...")
        print("=" * 60)

        close_el = next(
            (e for e in page.interactive_elements
             if e.icon_semantic == "close" and e.x > 800 and e.y < 150),
            None,
        )

        if close_el:
            ax, ay = close_el.x, close_el.y
            print(f"  胶囊× 识别坐标: ({ax:.0f}, {ay:.0f})")
        else:
            ax, ay = 945.0, 65.0
            print(f"  胶囊× 未识别，使用默认坐标: ({ax:.0f}, {ay:.0f})")

        lx, ly = logical_xy(ax, ay)
        print(f"  逻辑坐标: ({lx:.0f}, {ly:.0f})")

        input("\n  按回车执行关闭...")
        phone.client.tap(lx, ly)
        time.sleep(1.5)

        after = phone.screenshot()
        after_path = Path("/tmp/miniprogram_after_close.png")
        after_path.write_bytes(after)
        print(f"  关闭后截图: {after_path}")

        import subprocess
        subprocess.Popen(["open", str(after_path)])
        print("  完成。")


if __name__ == "__main__":
    main()

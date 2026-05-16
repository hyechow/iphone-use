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
from policy_expr.recon.page_parser import detect_miniprogram


def main() -> None:
    with LivePhoneSession() as phone:
        assert phone.client is not None

        print("\n" + "=" * 60)
        print("  截取当前页面...")
        print("=" * 60)
        png = phone.screenshot()

        # GUIClip capsule detection — returns close_xy or None
        close_xy = detect_miniprogram(png)
        print(f"\n  is_miniprogram: {close_xy is not None}")

        if close_xy is None:
            print("  未检测到小程序，退出。")
            return

        ax, ay = close_xy
        lx, ly = logical_xy(ax, ay)
        print(f"  close_xy: ({ax:.0f}, {ay:.0f})  →  逻辑({lx:.0f}, {ly:.0f})")

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

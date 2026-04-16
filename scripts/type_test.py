"""测试文字输入：点击输入框 → 粘贴文字 → 截图确认"""
import asyncio
import base64
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.utils import ocr_from_bytes, is_home_screen

# 先点击搜索框（主屏"搜索"），再输入文字
TAP_TARGET = "搜索"
INPUT_TEXT = "微信"


async def screenshot(session) -> bytes:
    result = await session.call_tool("screenshot", {})
    for item in result.content:
        if item.type == "image" and hasattr(item, "data"):
            return base64.b64decode(item.data)
    raise RuntimeError("截图失败")


async def main():
    server = StdioServerParameters(command="npx", args=["-y", "mirroir-mcp"])

    async with stdio_client(server) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            print("✅ 连接成功")

            # 1. 截图 + OCR
            print("📸 截图中...")
            png_bytes = await screenshot(session)
            ocr_results, (img_w, img_h) = ocr_from_bytes(png_bytes)
            win_w, win_h = img_w // 2, img_h // 2

            # 2. 找到输入框并点击
            home = is_home_screen(ocr_results)
            target = next((r for r in ocr_results if TAP_TARGET in r.text), None)
            if target is None:
                print(f"❌ 未找到 '{TAP_TARGET}'，当前文字：")
                for r in ocr_results:
                    print(f"   {r.text!r}")
                return

            y_offset = 0.06 if home else 0.0
            tx, ty = target.tap_coords(win_w, win_h, y_offset=y_offset)
            print(f"👆 点击 '{target.text}'  ({tx:.0f}, {ty:.0f})")
            tap_result = await session.call_tool("tap", {"x": tx, "y": ty})
            print("   结果:", tap_result.content)

            await asyncio.sleep(0.8)

            # 3. 写入剪贴板 + 用 osascript 发送 Cmd+V（支持中文）
            print(f"📋 写入剪贴板：{INPUT_TEXT!r}")
            subprocess.run(["pbcopy"], input=INPUT_TEXT.encode(), check=True)
            subprocess.run([
                "osascript", "-e",
                'tell application "System Events" to keystroke "v" using command down'
            ], check=True)
            print("   已粘贴")

            await asyncio.sleep(1)

            # 4. 截图确认
            print("📸 输入后截图...")
            after = await screenshot(session)
            out = Path(__file__).parent.parent / "screenshot_after.png"
            out.write_bytes(after)
            print(f"   已保存: {out}")


asyncio.run(main())

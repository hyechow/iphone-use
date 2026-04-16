"""测试 home 键和滑动手势"""
import asyncio
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def screenshot(session, name: str):
    result = await session.call_tool("screenshot", {})
    for item in result.content:
        if item.type == "image" and hasattr(item, "data"):
            out = Path(__file__).parent.parent / f"screenshot_{name}.png"
            out.write_bytes(base64.b64decode(item.data))
            print(f"   截图已保存: {out.name}")
            return


async def main():
    server = StdioServerParameters(command="npx", args=["-y", "mirroir-mcp"])

    async with stdio_client(server) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            print("✅ 连接成功\n")

            # --- 1. Home（tap 底部指示条，窗口高 701，y=685）---
            print("🏠 返回主屏（tap home indicator）...")
            result = await session.call_tool("tap", {"x": 159, "y": 685})
            print("   结果:", result.content)
            await asyncio.sleep(1)
            await screenshot(session, "home")

            # --- 2. 向上滑动（从底部往上，模拟返回主屏 / 滚动） ---
            print("\n👆 向上滑动（底部 → 中部）...")
            # 坐标为窗口像素，窗口大约 318×701
            result = await session.call_tool("swipe", {
                "from_x": 159, "from_y": 620,
                "to_x":   159, "to_y":   300,
                "duration_ms": 400,
            })
            print("   结果:", result.content)
            await asyncio.sleep(1)
            await screenshot(session, "swipe_up")

            # --- 3. 向下滑动（从顶部往下，模拟下拉通知栏） ---
            print("\n👇 向下滑动（顶部 → 中部）...")
            result = await session.call_tool("swipe", {
                "from_x": 159, "from_y": 10,
                "to_x":   159, "to_y":  350,
                "duration_ms": 400,
            })
            print("   结果:", result.content)
            await asyncio.sleep(1)
            await screenshot(session, "swipe_down")

            # --- 4. 再按 Home 回到主屏 ---
            print("\n🏠 再次按 Home 键回到主屏...")
            await session.call_tool("press_home", {})
            await asyncio.sleep(1)
            await screenshot(session, "final")


asyncio.run(main())

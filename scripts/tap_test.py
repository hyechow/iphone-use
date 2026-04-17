"""测试手机点击功能：截图 → OCR 找目标 → tap"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.utils import ocr_from_bytes, is_home_screen


TARGET = "微信"   # 要点击的文字，可以改成其他 App 名称


async def main():
    server = StdioServerParameters(command="npx", args=["-y", "mirroir-mcp"])

    async with stdio_client(server) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            print("✅ 连接成功")

            # 1. 截图
            print("📸 截图中...")
            import base64
            result = await session.call_tool("screenshot", {})
            png_bytes = None
            for item in result.content:
                if item.type == "image" and hasattr(item, "data"):
                    png_bytes = base64.b64decode(item.data)
                    break
            if not png_bytes:
                print("❌ 截图失败")
                return

            # 2. OCR 识别
            print("🔍 OCR 识别中...")
            ocr_results, (img_w, img_h) = ocr_from_bytes(png_bytes)
            # 截图为 Retina 2× 分辨率，tap 坐标用窗口逻辑像素
            win_w, win_h = img_w // 2, img_h // 2
            print(f"   识别到 {len(ocr_results)} 个元素，截图 {img_w}x{img_h}，窗口 {win_w}x{win_h}")

            # 3. 查找目标
            home = is_home_screen(ocr_results)
            target = next((r for r in ocr_results if TARGET in r.text), None)
            if target is None:
                print(f"❌ 未找到 '{TARGET}'，当前屏幕文字：")
                for r in ocr_results:
                    print(f"   [{r.confidence:.2f}] {r.text!r}")
                return

            # 主屏图标在文字标签上方，需向上偏移
            y_offset = 0.06 if home else 0.0
            print(f"   {'主屏' if home else 'App内'}模式，y_offset={y_offset}")
            tx, ty = target.tap_coords(win_w, win_h, y_offset=y_offset)
            print(f"✅ 找到 '{target.text}'  tap=({tx:.1f}, {ty:.1f})")

            # 4. 点击（像素坐标，y 轴已从 Vision 底左原点转换为顶左原点）
            print(f"👆 点击中...")
            tap_result = await session.call_tool("tap", {
                "x": tx,
                "y": ty,
            })
            print("   结果:", tap_result.content)

            # 5. 截图确认结果
            await asyncio.sleep(1)
            print("📸 点击后截图...")
            after = await session.call_tool("screenshot", {})
            for item in after.content:
                if item.type == "image" and hasattr(item, "data"):
                    out = Path(__file__).parent.parent / "images" / "screenshot_after.png"
                    out.write_bytes(base64.b64decode(item.data))
                    print(f"   已保存: {out}")
                    break


asyncio.run(main())

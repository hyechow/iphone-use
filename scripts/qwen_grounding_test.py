"""测试 Qwen3.5-35B-A3B 原生视觉定位能力（Visual Grounding）。

直接把截图图像发给模型，让它找到目标元素并返回点击坐标，
不依赖预处理的 OCR / 图标检测结果。

用法:
  # 离线模式（使用本地 images/screenshot.png）
  uv run python scripts/qwen_grounding_test.py "搜索框"
  uv run python scripts/qwen_grounding_test.py "微信图标"

  # 连接手机实时截图
  uv run python scripts/qwen_grounding_test.py --live "微信图标"
"""
import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from llm.provider_config import resolve_chat_provider_config

load_dotenv()

SCREENSHOT = Path(__file__).parent.parent / "images" / "screenshot.png"
OUTPUT = Path(__file__).parent.parent / "images" / "grounding_result.png"

cfg = resolve_chat_provider_config(provider="modelscope", model="Qwen/Qwen3.5-35B-A3B")
print(f"Provider : {cfg.provider}")
print(f"Model    : {cfg.model}")

SYSTEM_PROMPT = """\
你是一个 iPhone 操作助手。收到截图后：
1. 先用一两句话描述当前页面内容（是什么 App、什么界面）。
2. 输出简短可见依据：目标元素在哪里、为什么选择这个点击点。不要输出隐藏推理链。
3. 然后调用 tap_phone 工具点击目标元素（坐标系：左上角(0,0)，右下角(1000,1000)）。
"""

TAP_TOOL = {
    "type": "function",
    "function": {
        "name": "tap_phone",
        "description": "点击 iPhone 屏幕上的指定位置",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "归一化 x 坐标 (0-1000)"},
                "y": {"type": "number", "description": "归一化 y 坐标 (0-1000)"},
                "desc": {"type": "string", "description": "点击目标描述"},
            },
            "required": ["x", "y", "desc"],
        },
    },
}


def run_grounding(png_bytes: bytes, target: str) -> tuple[str | None, dict | None]:
    """把截图发给模型，返回 (页面描述, 归一化坐标)。"""
    import io
    img = Image.open(io.BytesIO(png_bytes))
    small = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"找到截图中的【{target}】并点击。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
        tools=[TAP_TOOL],
        tool_choice="auto",
        extra_body={"enable_thinking": False},
    )

    choice = resp.choices[0]
    description = choice.message.content or None
    coords = None
    if choice.message.tool_calls:
        tc = choice.message.tool_calls[0]
        args = json.loads(tc.function.arguments)
        # 防止模型返回 bbox list，取中点
        x = args.get("x", 0)
        y = args.get("y", 0)
        if isinstance(x, list):
            x = (x[0] + x[-1]) / 2
        if isinstance(y, list):
            y = (y[0] + y[-1]) / 2
        coords = {"x": x, "y": y, "desc": args.get("desc", "")}
    return description, coords



def visualize(png_bytes: bytes, x: float, y: float, desc: str, out_path: Path):
    img = Image.open(__import__("io").BytesIO(png_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 22)
    except Exception:
        font = ImageFont.load_default()

    # 逻辑像素 → 截图像素（Retina 2×）
    px, py = x * 2, y * 2
    R = 30
    draw.ellipse([px - R, py - R, px + R, py + R], fill=(255, 50, 50, 120))
    draw.ellipse([px - R, py - R, px + R, py + R], outline=(255, 50, 50, 255), width=3)
    draw.line([px - R - 8, py, px + R + 8, py], fill=(255, 50, 50, 255), width=3)
    draw.line([px, py - R - 8, px, py + R + 8], fill=(255, 50, 50, 255), width=3)
    draw.text((px + R + 8, py - 14), f"{desc} ({x:.0f},{y:.0f})", fill=(255, 50, 50, 255), font=font)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(out_path)
    print(f"🖼️  可视化已保存: {out_path}")


# ── 离线模式 ─────────────────────────────────────────────────────────────────

def offline_mode(target: str):
    if not SCREENSHOT.exists():
        print(f"截图不存在: {SCREENSHOT}")
        print("请先运行 scripts/screenshot_test.py 或使用 --live 模式")
        sys.exit(1)

    png_bytes = SCREENSHOT.read_bytes()
    img = Image.open(__import__("io").BytesIO(png_bytes))
    print(f"截图尺寸: {img.size[0]}x{img.size[1]}（逻辑像素 {img.size[0]//2}x{img.size[1]//2}）")

    print(f"\n🤖 发送截图给模型，目标: 【{target}】 ...")
    description, coords = run_grounding(png_bytes, target)
    if description:
        print(f"\n📋 页面描述: {description}")
    if coords is None:
        print("⚠️ 模型未返回坐标")
        return

    nx, ny = coords.get("x", 0), coords.get("y", 0)
    desc = coords.get("desc", target)
    x, y = nx / 1000 * 318, ny / 1000 * 701
    print(f"✅ 归一化坐标: ({nx}, {ny})  →  逻辑像素: ({x:.0f}, {y:.0f})  {desc}")
    visualize(png_bytes, x, y, desc, OUTPUT)


# ── 连接手机模式 ──────────────────────────────────────────────────────────────

async def live_mode(target: str):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from contextlib import AsyncExitStack

    print("📱 连接手机中...")
    server = StdioServerParameters(command="npx", args=["-y", "mirroir-mcp"])
    async with AsyncExitStack() as stack:
        r, w = await stack.enter_async_context(stdio_client(server))
        session = await stack.enter_async_context(ClientSession(r, w))
        await session.initialize()
        print("✅ MCP 连接成功")

        result = await session.call_tool("screenshot", {})
        png_bytes = None
        for item in result.content:
            if item.type == "image" and hasattr(item, "data"):
                png_bytes = base64.b64decode(item.data)
                break
        if not png_bytes:
            print("❌ 截图失败")
            return

        SCREENSHOT.write_bytes(png_bytes)
        img = Image.open(__import__("io").BytesIO(png_bytes))
        print(f"📸 截图成功: {img.size[0]}x{img.size[1]}（逻辑像素 {img.size[0]//2}x{img.size[1]//2}）")

        print(f"\n🤖 发送截图给模型，目标: 【{target}】 ...")
        description, coords = run_grounding(png_bytes, target)
        if description:
            print(f"\n📋 页面描述: {description}")
        if coords is None:
            print("⚠️ 模型未返回坐标")
            return

        nx, ny = coords.get("x", 0), coords.get("y", 0)
        desc = coords.get("desc", target)
        x, y = nx / 1000 * 318, ny / 1000 * 701
        print(f"✅ 归一化坐标: ({nx}, {ny})  →  逻辑像素: ({x:.0f}, {y:.0f})  {desc}")
        visualize(png_bytes, x, y, desc, OUTPUT)

        print(f"\n👆 执行点击: ({x}, {y}) - {desc}")
        tap_result = await session.call_tool("tap", {"x": x, "y": y})
        print(f"   MCP 结果: {tap_result.content}")

        await asyncio.sleep(1.5)
        after = await session.call_tool("screenshot", {})
        for item in after.content:
            if item.type == "image" and hasattr(item, "data"):
                after_bytes = base64.b64decode(item.data)
                after_path = Path(__file__).parent.parent / "images" / "screenshot_after.png"
                after_path.write_bytes(after_bytes)
                print(f"📸 点击后截图: {after_path}")
                break


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="测试 Qwen3.5-35B-A3B 视觉定位能力")
    parser.add_argument("target", nargs="?", default="搜索框", help="目标元素描述")
    parser.add_argument("--live", action="store_true", help="连接手机实时截图并执行点击")
    args = parser.parse_args()

    if args.live:
        asyncio.run(live_mode(args.target))
    else:
        offline_mode(args.target)


if __name__ == "__main__":
    main()

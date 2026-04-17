"""测试 LLM 对 UI 截图的推理能力：融合 OCR + Icon 检测，让 LLM 决定点击位置并可视化。

用法:
  # 离线模式（仅推理 + 可视化，需要本地 screenshot.png）
  uv run python scripts/llm_ui_reasoning_test.py "打开微信"

  # 连接手机模式（截图 + 推理 + 执行点击 + 点击后截图确认）
  uv run python scripts/llm_ui_reasoning_test.py --live "打开微信"
"""
import argparse
import asyncio
import base64
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from PIL import Image, ImageDraw, ImageFont

from agent.utils import detect_icons, ocr_from_bytes
from llm.provider_config import resolve_chat_provider_config

load_dotenv()

SCREENSHOT = Path(__file__).parent.parent / "images" / "screenshot.png"
OUTPUT = Path(__file__).parent.parent / "images" / "llm_ui_result.png"

# ── LLM 配置 ────────────────────────────────────────────────────────────────

cfg = resolve_chat_provider_config(provider="modelscope", model="Qwen/Qwen3.5-35B-A3B")
print(f"Provider : {cfg.provider}")
print(f"Model    : {cfg.model}")
print(f"Base URL : {cfg.base_url}")

SYSTEM_PROMPT = """\
你是一个 iPhone 操作助手。用户会告诉你他想做什么，你需要根据屏幕上的 OCR 文字和图标检测结果，决定要点击的位置。

屏幕信息格式：
- 窗口大小：win_w x win_h（逻辑像素）
- OCR 结果：文字、置信度、中心位置（逻辑像素坐标）
- 图标检测结果：置信度、中心位置和 bbox（逻辑像素坐标）

你会收到一个 tap_phone 工具，参数为点击坐标 x, y（逻辑像素，左上角为原点，范围 0~win_w, 0~win_h）。
请分析屏幕内容，找到最符合用户意图的目标，调用 tap_phone 工具执行点击。
注意：在主屏幕上点击 App 图标时，图标在文字标签上方，应该点击图标的中心位置（而不是文字标签的位置）。
"""


@tool
def tap_phone(x: float, y: float, target_desc: str) -> str:
    """点击 iPhone 屏幕上的指定位置。

    Args:
        x: 点击位置的 x 坐标（逻辑像素，左上角为原点）
        y: 点击位置的 y 坐标（逻辑像素，左上角为原点）
        target_desc: 点击目标的描述（如"微信图标"、"搜索框"）
    """
    return f"已点击 ({x}, {y})，目标: {target_desc}"


# ── MCP 连接（仅 --live 模式使用） ──────────────────────────────────────────

async def mcp_screenshot(session) -> bytes:
    from mcp import ClientSession
    result = await session.call_tool("screenshot", {})
    for item in result.content:
        if item.type == "image" and hasattr(item, "data"):
            return base64.b64decode(item.data)
    raise RuntimeError("截图失败")


async def mcp_tap(session, x: float, y: float):
    result = await session.call_tool("tap", {"x": x, "y": y})
    print(f"   MCP 结果: {result.content}")


async def mcp_connect():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    stack = AsyncExitStack()
    server = StdioServerParameters(command="npx", args=["-y", "mirroir-mcp"])
    r, w = await stack.enter_async_context(stdio_client(server))
    session = await stack.enter_async_context(ClientSession(r, w))
    await session.initialize()
    return stack, session


# ── 可视化 ──────────────────────────────────────────────────────────────────

def visualize(img_path: Path, ocr_results, icons, win_w, win_h, tool_calls, out_path: Path):
    """将 OCR、图标检测和 LLM tool call 结果可视化到一张图上。"""
    img = Image.open(img_path).convert("RGBA")
    img_w, img_h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 16)
    except Exception:
        font = font_small = ImageFont.load_default()

    # 图标 bbox（蓝色）
    for icon in icons:
        draw.rectangle(
            [icon.x1, icon.y1, icon.x2, icon.y2],
            outline=(0, 150, 255, 140), width=2,
        )

    # OCR 文字（绿色）
    for r in ocr_results:
        x1 = r.x * img_w
        y1 = (1.0 - r.y - r.height) * img_h
        x2 = (r.x + r.width) * img_w
        y2 = (1.0 - r.y) * img_h
        draw.rectangle([x1, y1, x2, y2], outline=(80, 200, 80, 120), width=2)
        draw.text((x1 + 2, y2 + 2), r.text, fill=(80, 200, 80, 200), font=font_small)

    # LLM tool call 点击位置（红色大圆 + 十字线）
    for tc in tool_calls:
        if tc["name"] != "tap_phone":
            continue
        x, y = tc["args"]["x"], tc["args"]["y"]
        desc = tc["args"].get("target_desc", "")
        # 逻辑像素 → 截图像素 (2x)
        px, py = x * 2, y * 2
        R = 30
        draw.ellipse([px - R, py - R, px + R, py + R], fill=(255, 0, 0, 100))
        draw.ellipse([px - R, py - R, px + R, py + R], outline=(255, 0, 0, 255), width=3)
        draw.line([px - R - 5, py, px + R + 5, py], fill=(255, 0, 0, 255), width=3)
        draw.line([px, py - R - 5, px, py + R + 5], fill=(255, 0, 0, 255), width=3)
        draw.text((px + R + 6, py - 12), f"{desc} ({x:.0f},{y:.0f})", fill=(255, 0, 0, 255), font=font)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(out_path)


# ── 构建 LLM 输入 ──────────────────────────────────────────────────────────

def build_screen_desc(ocr_results, icons, win_w, win_h):
    lines = []
    lines.append(f"## 屏幕信息 (窗口 {win_w}x{win_h} 逻辑像素)\n")

    lines.append("### OCR 检测结果\n")
    if ocr_results:
        lines.append("| 文字 | 置信度 | 中心位置(逻辑像素) |")
        lines.append("|------|--------|---------------------|")
        for r in ocr_results:
            cx, cy = r.center_x * win_w, (1 - r.center_y) * win_h
            lines.append(f"| {r.text} | {r.confidence:.2f} | ({cx:.0f}, {cy:.0f}) |")
    else:
        lines.append("(无)")

    lines.append("\n### 图标检测结果\n")
    if icons:
        lines.append("| # | 置信度 | 中心位置(逻辑像素) | bbox(逻辑像素) |")
        lines.append("|---|--------|---------------------|-----------------|")
        for i, icon in enumerate(icons):
            cx, cy = icon.center[0] / 2, icon.center[1] / 2
            bbox = f"[{icon.x1/2:.0f},{icon.y1/2:.0f},{icon.x2/2:.0f},{icon.y2/2:.0f}]"
            lines.append(
                f"| {i+1} | {icon.confidence:.2f} | ({cx:.0f}, {cy:.0f}) | {bbox} |"
            )
    else:
        lines.append("(无)")

    return "\n".join(lines)


# ── LLM 推理 ────────────────────────────────────────────────────────────────

def run_llm(screen_desc: str, user_input: str):
    llm = ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    ).bind_tools([tap_phone])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"{screen_desc}\n\n## 用户指令\n\n{user_input}"),
    ]
    return llm.invoke(messages)


# ── 离线模式 ────────────────────────────────────────────────────────────────

def offline_mode(user_input: str):
    if not SCREENSHOT.exists():
        print("screenshot.png 不存在，请先运行 scripts/screenshot_test.py 或使用 --live 模式")
        sys.exit(1)

    png_bytes = SCREENSHOT.read_bytes()

    print("🔍 OCR 识别中...")
    ocr_results, (img_w, img_h) = ocr_from_bytes(png_bytes)
    win_w, win_h = img_w // 2, img_h // 2
    print(f"   识别到 {len(ocr_results)} 个文字元素")

    print("🎯 图标检测中...")
    icons = detect_icons(png_bytes, conf=0.3)
    print(f"   检测到 {len(icons)} 个图标")

    screen_desc = build_screen_desc(ocr_results, icons, win_w, win_h)

    print(f"\n🤖 LLM 推理中...")
    print(f"   screen_desc: {screen_desc}")
    response = run_llm(screen_desc, user_input)
    print(f"   回复: {response.content}")

    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"\n🔧 Tool Call: {tc['name']}")
            print(f"   参数: {json.dumps(tc['args'], ensure_ascii=False)}")
        visualize(SCREENSHOT, ocr_results, icons, win_w, win_h, response.tool_calls, OUTPUT)
        print(f"\n🖼️  可视化已保存: {OUTPUT}")
    else:
        print("\n⚠️ LLM 没有生成 tool call")


# ── 连接手机模式 ────────────────────────────────────────────────────────────

async def live_mode(user_input: str):
    print("📱 连接手机中...")
    stack, session = await mcp_connect()
    print("✅ MCP 连接成功")

    try:
        await _live_run(session, user_input)
    finally:
        await stack.aclose()


async def _live_run(session, user_input: str):
    # 1. 截图
    print("\n📸 截图中...")
    png_bytes = await mcp_screenshot(session)
    SCREENSHOT.write_bytes(png_bytes)
    print(f"   截图大小: {len(png_bytes)//1024} KB")

    # 2. OCR + Icon
    ocr_results, (img_w, img_h) = ocr_from_bytes(png_bytes)
    win_w, win_h = img_w // 2, img_h // 2
    print(f"🔍 OCR: {len(ocr_results)} 个文字元素")

    icons = detect_icons(png_bytes, conf=0.3)
    print(f"🎯 图标: {len(icons)} 个图标")

    # 3. LLM 推理
    screen_desc = build_screen_desc(ocr_results, icons, win_w, win_h)

    print(f"\n🤖 LLM 推理中...")
    print(f"   screen_desc: {screen_desc}")
    response = run_llm(screen_desc, user_input)
    print(f"   回复: {response.content}")

    if not response.tool_calls:
        print("\n⚠️ LLM 没有生成 tool call")
        return

    for tc in response.tool_calls:
        print(f"\n🔧 Tool Call: {tc['name']}")
        print(f"   参数: {json.dumps(tc['args'], ensure_ascii=False)}")

        if tc["name"] == "tap_phone":
            x, y = tc["args"]["x"], tc["args"]["y"]
            desc = tc["args"].get("target_desc", "")

            # 可视化（点击前）
            visualize(SCREENSHOT, ocr_results, icons, win_w, win_h, response.tool_calls, OUTPUT)
            print(f"\n🖼️  可视化已保存: {OUTPUT}")

            # 执行点击
            print(f"\n👆 执行点击: ({x:.0f}, {y:.0f}) - {desc}")
            await mcp_tap(session, x, y)

            # 点击后截图确认
            await asyncio.sleep(1.5)
            print("\n📸 点击后截图...")
            after_bytes = await mcp_screenshot(session)
            after_path = Path(__file__).parent.parent / "images" / "screenshot_after.png"
            after_path.write_bytes(after_bytes)
            print(f"   已保存: {after_path}")


# ── 入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="测试 LLM 对 UI 截图的推理能力")
    parser.add_argument("instruction", nargs="?", default="打开微信", help="操作指令")
    parser.add_argument("--live", action="store_true", help="连接手机模式（截图 + 执行点击）")
    args = parser.parse_args()

    if args.live:
        asyncio.run(live_mode(args.instruction))
    else:
        offline_mode(args.instruction)


if __name__ == "__main__":
    main()

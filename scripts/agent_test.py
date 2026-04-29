"""Agent 模拟测试：输入截图文件 + 自然语言指令，输出 Action 可视化。

用法:
  uv run python scripts/agent_test.py images/home.png "打开微信"
  uv run python scripts/agent_test.py images/home.png "打开微信" -o result.png
"""
import argparse
import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict

from agent.context import ContextBuilder
from agent.logger import LLMLogger
from llm.provider_config import resolve_chat_provider_config

load_dotenv()

# ── Actions recorder ───────────────────────────────────────────────────────────

class Action:
    def __init__(self, action_type: str, params: dict):
        self.type = action_type
        self.params = params

    def __repr__(self):
        return f"Action({self.type}, {self.params})"


_actions: list[Action] = []
_screenshot_b64: str = ""


def load_screenshot(path: str):
    global _screenshot_b64
    img = Image.open(path)
    if img.width > 500:
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    _screenshot_b64 = base64.b64encode(buf.getvalue()).decode()


# ── Mock tools ─────────────────────────────────────────────────────────────────

@tool
def take_screenshot(config: RunnableConfig) -> str:
    """Take a screenshot of the current iPhone screen."""
    _actions.append(Action("screenshot", {}))
    return _screenshot_b64


@tool
def tap_screen(x: float, y: float, config: RunnableConfig) -> str:
    """Tap a position on the iPhone screen.
    Args:
        x: Normalized x coordinate (0-1000, left=0, right=1000)
        y: Normalized y coordinate (0-1000, top=0, bottom=1000)
    """
    _actions.append(Action("tap", {"x": x, "y": y}))
    return f"Tapped at ({x:.0f}, {y:.0f})"


@tool
def go_to_home_screen(config: RunnableConfig) -> str:
    """Return to the iPhone home screen."""
    _actions.append(Action("home", {}))
    return "Tapped home indicator"


@tool
def type_text(text: str, config: RunnableConfig) -> str:
    """Type text into the currently focused input field on the iPhone."""
    _actions.append(Action("type", {"text": text}))
    return f"Typed: {text!r}"


MOCK_TOOLS = [take_screenshot, tap_screen, go_to_home_screen, type_text]


# ── Agent graph ────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


_cfg = resolve_chat_provider_config(provider="modelscope", model="Qwen/Qwen3.5-35B-A3B")
_llm = ChatOpenAI(
    model=_cfg.model, api_key=_cfg.api_key, base_url=_cfg.base_url,
).bind_tools(MOCK_TOOLS)

_system = SystemMessage(content=(
    "你是一个 iPhone 操作助手。需要查看屏幕时，调用 take_screenshot 工具获取截图，"
    "然后根据截图内容和用户指令给出操作。"
    "坐标系：左上角(0,0)，右下角(1000,1000)。"
))

_context_builder = ContextBuilder(_system)
_llm_logger = LLMLogger()  # reconfigured in main()


def agent_node(state: State) -> dict:
    context = _context_builder.build(state["messages"])
    response = _llm.invoke(context)
    _llm_logger.log(context, response)
    return {"messages": [response]}


def _build_graph():
    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(MOCK_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=MemorySaver())


# ── Visualization ──────────────────────────────────────────────────────────────

_COLORS = {
    "screenshot": "#4285f4",
    "tap":        "#ea4335",
    "home":       "#34a853",
    "type":       "#f0a500",
}

def visualize(screenshot_path: str, output_path: str):
    img = Image.open(screenshot_path).copy().convert("RGBA")
    if img.width > 500:
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    tap_count = 0
    for i, action in enumerate(_actions):
        color = _COLORS.get(action.type, "#ffffff")
        label = f"Step {i+1}"

        if action.type == "tap":
            tap_count += 1
            x = int(action.params["x"] / 1000 * w)
            y = int(action.params["y"] / 1000 * h)
            r = 24
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)
            draw.text((x + r + 6, y - 10), f"{label} Tap({action.params['x']:.0f},{action.params['y']:.0f})",
                      fill=color, font=font)
        elif action.type == "home":
            bx, by = w // 2, h - 16
            draw.ellipse([bx - 30, by - 30, bx + 30, by + 30], outline=color, width=3)
            draw.text((bx + 36, by - 10), f"{label} Home", fill=color, font=font)
        elif action.type == "type":
            text = action.params["text"]
            draw.text((16, h - 50 - tap_count * 30), f"{label} Type: {text}", fill=color, font=font)
        elif action.type == "screenshot":
            draw.text((16, 12), f"{label} Screenshot", fill=color, font=font_sm)

    img.save(output_path)
    print(f"\n可视化结果已保存: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_test(screenshot_path: str, instruction: str, output_path: str, max_steps: int = 10):
    global _actions
    _actions.clear()

    print(f"截图: {screenshot_path}")
    print(f"指令: {instruction}")

    load_screenshot(screenshot_path)
    graph = _build_graph()
    config = {"configurable": {"thread_id": "test"}}

    input_state = {
        "messages": [HumanMessage(content=[
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_screenshot_b64}"}},
        ])],
    }

    step = 0
    for item in graph.stream(input_state, config, stream_mode=["messages", "updates"], version="v2"):
        if item["type"] == "updates":
            for node_name, update in item["data"].items():
                if node_name == "agent":
                    step += 1
                elif node_name == "tools":
                    for msg in update.get("messages", []):
                        if hasattr(msg, "name") and msg.name:
                            _actions.append(Action(msg.name, getattr(msg, "args", {})))
                    if step >= max_steps:
                        print(f"\n达到最大步数 ({max_steps})，停止")
                        return

    print(f"\n共执行 {len(_actions)} 步: {_actions}")
    visualize(screenshot_path, output_path)


def main():
    global _llm_logger

    parser = argparse.ArgumentParser(description="Agent 模拟测试 - Action 可视化")
    parser.add_argument("screenshot", help="UI 截图文件路径")
    parser.add_argument("instruction", help="自然语言指令")
    parser.add_argument("-o", "--output", default=None, help="输出图片路径（默认: <screenshot>_result.png）")
    parser.add_argument("--log-dir", default=None, help="LLM 调用日志目录（写入 llm_calls.jsonl）")
    parser.add_argument("--max-steps", type=int, default=10, help="最大执行步数（默认: 3）")
    args = parser.parse_args()

    output = args.output or str(Path(args.screenshot).with_name(Path(args.screenshot).stem + "_result.png"))
    _llm_logger = LLMLogger(log_dir=args.log_dir)

    run_test(args.screenshot, args.instruction, output, args.max_steps)


if __name__ == "__main__":
    main()

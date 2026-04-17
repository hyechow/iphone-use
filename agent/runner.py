import asyncio
import base64
import io
import time

from langchain_core.messages import HumanMessage
from PIL import Image

from agent.agent import _graph
from agent.events import AgentEvent
from agent.sessions import close_session, get_client


def _ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


async def _take_initial_screenshot(thread_id: str) -> str:
    """Take a screenshot via MCP and return resized base64 PNG."""
    client = await get_client(thread_id)
    png_bytes = await client.screenshot()
    img = Image.open(io.BytesIO(png_bytes))
    small = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class PhoneAgent:
    async def run(self, thread_id: str, instruction: str, queue: asyncio.Queue[AgentEvent]):
        config = {"configurable": {"thread_id": thread_id}}

        print(f"{_ts()} runner: taking initial screenshot", flush=True)
        b64 = await _take_initial_screenshot(thread_id)
        await queue.put(AgentEvent(type="screenshot", data=b64))
        print(f"{_ts()} runner: screenshot ready ({len(b64)} bytes base64)", flush=True)

        input_state = {
            "messages": [HumanMessage(content=[
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ])],
        }
        print(f"{_ts()} runner: starting graph stream", flush=True)

        try:
            async for event in _graph.astream_events(input_state, config, version="v2"):
                kind = event["event"]
                name = event.get("name", "")
                # print(f"{_ts()} graph event: {kind} name={name!r}", flush=True)

                # 工具调用完成：截图结果推给前端
                if kind == "on_tool_end" and name == "take_screenshot":
                    output = event["data"].get("output", "")
                    b64 = output.content if hasattr(output, "content") else output
                    if b64:
                        await queue.put(AgentEvent(type="screenshot", data=b64))

                # LLM token 流
                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    token = chunk.content
                    if isinstance(token, str) and token:
                        await queue.put(AgentEvent(type="thinking", data=token))
                    elif isinstance(token, list):
                        for part in token:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text = part.get("text", "")
                                if text:
                                    await queue.put(AgentEvent(type="thinking", data=text))

            print(f"{_ts()} runner: graph stream finished", flush=True)
            await queue.put(AgentEvent(type="done", data="完成"))

        except Exception as e:
            print(f"{_ts()} runner: exception {e!r}", flush=True)
            await queue.put(AgentEvent(type="error", data=str(e)))
        finally:
            await close_session(thread_id)

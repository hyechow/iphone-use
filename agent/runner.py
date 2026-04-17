import base64
import io

from langchain_core.messages import HumanMessage, ToolMessage
from PIL import Image

from agent.agent import _graph
from agent.events import AgentEvent
from agent.sessions import close_session, get_client


def _take_initial_screenshot(thread_id: str) -> str:
    """Take a screenshot and return resized base64 PNG."""
    client = get_client(thread_id)
    png_bytes = client.screenshot()
    img = Image.open(io.BytesIO(png_bytes))
    small = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class PhoneAgent:
    def run(self, thread_id: str, instruction: str):
        """Run agent synchronously. Yields AgentEvent in real-time."""
        config = {"configurable": {"thread_id": thread_id}}

        b64 = _take_initial_screenshot(thread_id)
        yield AgentEvent(type="screenshot", data=b64)

        input_state = {
            "messages": [HumanMessage(content=[
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ])],
        }

        try:
            in_think = False
            think_buf = ""

            for item in _graph.stream(input_state, config, stream_mode=["messages", "updates"], version="v2"):
                mode = item["type"]
                data = item["data"]

                if mode == "messages":
                    msg, meta = data
                    if meta.get("langgraph_node") != "agent":
                        continue
                    token = msg.content if isinstance(msg.content, str) else None
                    if not token:
                        continue

                    # Parse <think...</think blocks into reasoning events
                    if in_think:
                        end_idx = token.find("</think")
                        if end_idx >= 0:
                            think_buf += token[:end_idx]
                            if think_buf.strip():
                                yield AgentEvent(type="reasoning", data=think_buf)
                            think_buf = ""
                            token = token[end_idx + 8:]
                            in_think = False
                            if token.strip():
                                yield AgentEvent(type="thinking", data=token)
                        else:
                            think_buf += token
                    else:
                        start_idx = token.find("<think")
                        if start_idx >= 0:
                            if start_idx > 0 and token[:start_idx].strip():
                                yield AgentEvent(type="thinking", data=token[:start_idx])
                            rest = token[start_idx + 6:]
                            gt = rest.find(">")
                            rest = rest[gt + 1:] if gt >= 0 else ""
                            end_idx = rest.find("</think")
                            if end_idx >= 0:
                                if rest[:end_idx].strip():
                                    yield AgentEvent(type="reasoning", data=rest[:end_idx])
                                rest = rest[end_idx + 8:]
                                if rest.strip():
                                    yield AgentEvent(type="thinking", data=rest)
                            else:
                                think_buf = rest
                                in_think = True
                        else:
                            yield AgentEvent(type="thinking", data=token)

                elif mode == "updates":
                    for node_name, update in data.items():
                        if node_name == "tools":
                            for msg in update.get("messages", []):
                                if isinstance(msg, ToolMessage) and msg.name == "take_screenshot":
                                    b64 = msg.content
                                    if isinstance(b64, str) and len(b64) > 100:
                                        yield AgentEvent(type="screenshot", data=b64)

            if in_think and think_buf.strip():
                yield AgentEvent(type="reasoning", data=think_buf)

            yield AgentEvent(type="done", data="完成")

        except Exception as e:
            yield AgentEvent(type="error", data=str(e))
        finally:
            close_session(thread_id)

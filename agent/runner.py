import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent.agent import _graph
from agent.events import AgentEvent
from agent.limits import MAX_REACT_TOOL_ROUNDS, REACT_RECURSION_LIMIT
from agent.sessions import close_session
from agent.visualizer import ReActVisualizer

_TURN_COUNTS: dict[str, int] = {}


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def _next_turn_id(thread_id: str) -> str:
    """Return the next user-agent turn number for a thread."""
    safe_thread_id = _safe_path_part(thread_id)
    if thread_id not in _TURN_COUNTS:
        existing_turns = []
        thread_log_dir = Path("logs") / safe_thread_id
        if thread_log_dir.exists():
            for path in thread_log_dir.iterdir():
                if path.is_dir() and path.name.isdigit():
                    existing_turns.append(int(path.name))
        _TURN_COUNTS[thread_id] = max(existing_turns, default=0)

    _TURN_COUNTS[thread_id] += 1
    return str(_TURN_COUNTS[thread_id])


class PhoneAgent:
    def run(self, thread_id: str, instruction: str):
        """Run agent synchronously. Yields AgentEvent in real-time."""
        turn_id = _next_turn_id(thread_id)
        visualizer = ReActVisualizer(thread_id, turn_id)
        config = {
            "configurable": {"thread_id": thread_id},
            # One ReAct round is roughly agent -> tools, plus the initial agent node.
            "recursion_limit": REACT_RECURSION_LIMIT,
        }

        input_state = {
            "messages": [HumanMessage(content=[
                {"type": "text", "text": instruction},
            ])],
        }

        try:
            in_think = False
            think_buf = ""
            latest_screenshot_b64: str | None = None
            pending_actions: dict[str, dict] = {}
            pending_action_ids: list[str] = []

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
                        else:
                            think_buf += token
                    else:
                        start_idx = token.find("<think")
                        if start_idx >= 0:
                            rest = token[start_idx + 6:]
                            gt = rest.find(">")
                            rest = rest[gt + 1:] if gt >= 0 else ""
                            end_idx = rest.find("</think")
                            if end_idx >= 0:
                                if rest[:end_idx].strip():
                                    yield AgentEvent(type="reasoning", data=rest[:end_idx])
                            else:
                                think_buf = rest
                                in_think = True

                elif mode == "updates":
                    for node_name, update in data.items():
                        if node_name == "agent":
                            b64 = update.get("latest_screenshot")
                            if isinstance(b64, str) and len(b64) > 100:
                                latest_screenshot_b64 = b64
                                visualizer.update_screenshot(b64)
                                yield AgentEvent(type="screenshot", data=b64)
                            for msg in update.get("messages", []):
                                if isinstance(msg, AIMessage):
                                    for tool_call in msg.tool_calls:
                                        action_name = tool_call.get("name", "unknown")
                                        action_id = tool_call.get("id") or f"pending-{len(pending_action_ids) + 1}"
                                        pending_actions[action_id] = {
                                            "name": action_name,
                                            "args": tool_call.get("args", {}),
                                            "screenshot_b64": latest_screenshot_b64,
                                        }
                                        pending_action_ids.append(action_id)
                        elif node_name == "tools":
                            for msg in update.get("messages", []):
                                if not isinstance(msg, ToolMessage):
                                    continue

                                action_id = getattr(msg, "tool_call_id", None)
                                action = pending_actions.pop(action_id, None) if action_id else None
                                if action_id in pending_action_ids:
                                    pending_action_ids.remove(action_id)
                                if action is None and pending_action_ids:
                                    fallback_id = pending_action_ids.pop(0)
                                    action = pending_actions.pop(fallback_id, None)

                                if action:
                                    action_name = action["name"]
                                    try:
                                        action_path = visualizer.save_action(
                                            action_name,
                                            action["args"],
                                            screenshot_b64=action["screenshot_b64"],
                                        )
                                    except Exception as e:
                                        yield AgentEvent(
                                            type="action",
                                            data=f"{action_name} visualization failed: {e}",
                                        )
                                        action_path = None
                                    if action_path:
                                        yield AgentEvent(
                                            type="action",
                                            data=f"{action_name} -> {action_path}",
                                        )

                                if msg.name == "take_screenshot":
                                    b64 = msg.content
                                    if isinstance(b64, str) and len(b64) > 100:
                                        latest_screenshot_b64 = b64
                                        visualizer.update_screenshot(b64)
                                        yield AgentEvent(type="screenshot", data=b64)

            if in_think and think_buf.strip():
                yield AgentEvent(type="reasoning", data=think_buf)

            yield AgentEvent(type="done", data="完成")

        except GraphRecursionError:
            yield AgentEvent(
                type="error",
                data=f"已达到最大操作轮数（{MAX_REACT_TOOL_ROUNDS} 轮），停止执行",
            )
        except Exception as e:
            yield AgentEvent(type="error", data=str(e))
        finally:
            close_session(thread_id)

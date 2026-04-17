from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from agent.tools import TOOLS
from llm.provider_config import resolve_chat_provider_config

load_dotenv()

# ── LangGraph State ───────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ── Graph Nodes ───────────────────────────────────────────────────────────────

_cfg = resolve_chat_provider_config(provider="modelscope", model="Qwen/Qwen3.5-35B-A3B")
print(f"Agent LLM: {_cfg.provider} / {_cfg.model}")
_llm = ChatOpenAI(
    model=_cfg.model,
    api_key=_cfg.api_key,
    base_url=_cfg.base_url,
).bind_tools(TOOLS)

_system = SystemMessage(content=(
    "你是一个 iPhone 操作助手。需要查看屏幕时，调用 take_screenshot 工具获取截图，"
    "然后根据截图内容和用户指令给出操作。"
    "坐标系：左上角(0,0)，右下角(1000,1000)。"
))


def _inject_screenshot_images(messages: list) -> list:
    """Convert take_screenshot ToolMessage base64 content to image_url format.

    The ToolNode returns raw base64 text, but Qwen vision requires image_url.
    We replace each such ToolMessage with a HumanMessage containing the image.
    """
    result = []
    for msg in messages:
        if (
            isinstance(msg, ToolMessage)
            and msg.name == "take_screenshot"
            and isinstance(msg.content, str)
            and len(msg.content) > 100  # likely base64
        ):
            result.append(HumanMessage(content=[
                {"type": "text", "text": "[截图结果]"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{msg.content}"},
                },
            ]))
        else:
            result.append(msg)
    return result


async def agent_node(state: State) -> dict:
    messages = _inject_screenshot_images(state["messages"])
    response = await _llm.ainvoke([_system] + messages)
    return {"messages": [response]}


# ── Build Graph ───────────────────────────────────────────────────────────────

_checkpointer = MemorySaver()


def _build_graph():
    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=_checkpointer)


_graph = _build_graph()

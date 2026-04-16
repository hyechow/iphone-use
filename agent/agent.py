from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
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

_cfg = resolve_chat_provider_config()
_llm = ChatOpenAI(
    model=_cfg.model,
    api_key=_cfg.api_key,
    base_url=_cfg.base_url,
).bind_tools(TOOLS)

_system = SystemMessage(content=(
    "你是一个 iPhone 操作助手。需要查看屏幕时，调用 take_screenshot 工具获取截图，"
    "然后根据截图内容和用户指令给出操作建议。"
))


async def agent_node(state: State) -> dict:
    response = await _llm.ainvoke([_system] + state["messages"])
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
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=_checkpointer)


_graph = _build_graph()

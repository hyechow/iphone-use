from dataclasses import dataclass
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from agent.tools import TOOLS


@dataclass
class AgentEvent:
    type: str  # "screenshot" | "thinking" | "action" | "done" | "error"
    data: str  # text, or base64-encoded PNG for "screenshot"


# ── LangGraph State ───────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ── Graph Nodes ───────────────────────────────────────────────────────────────

_llm = ChatAnthropic(model="claude-opus-4-6", max_tokens=1024).bind_tools(TOOLS)

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

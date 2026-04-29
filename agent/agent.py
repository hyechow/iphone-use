import time
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import NotRequired, TypedDict

from agent.context import ContextBuilder
from agent.limits import MAX_REACT_TOOL_ROUNDS, POST_TOOL_SCREEN_SETTLE_SECONDS
from agent.logger import LLMLogger
from agent.tool_args import normalize_tool_args
from agent.tools import TOOLS, take_screenshot
from llm.provider_config import resolve_chat_provider_config

load_dotenv()

# ── LangGraph State ───────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    latest_screenshot: NotRequired[str]
    tool_rounds: NotRequired[int]


# ── LLM + context + logger ────────────────────────────────────────────────────

_cfg = resolve_chat_provider_config(provider="modelscope", model="Qwen/Qwen3.5-35B-A3B")
_llm = ChatOpenAI(
    model=_cfg.model,
    api_key=_cfg.api_key,
    base_url=_cfg.base_url,
    extra_body={"enable_thinking": False},
).bind_tools(TOOLS, strict=True, parallel_tool_calls=False)

_system = SystemMessage(content=(
    "你是一个 iPhone 操作助手。每轮用户指令都会附带当前屏幕截图，"
    "你必须根据截图内容和用户指令执行操作。"
    "如果用户请求需要点击、输入、返回主页或打开页面，你不能只输出文字，"
    "必须调用一个合适的工具。"
    "每次调用工具前，可以先用一两句话输出简短可见操作依据："
    "你观察到了什么目标、为什么选择这个动作；随后必须立即调用工具。"
    "只有当任务已经完成、无需任何手机操作时，才可以不调用工具。"
    "坐标系：左上角(0,0)，右下角(1000,1000)。"
    "调用 tap_screen 时，x 和 y 必须是单个数字，表示点击点中心；"
    "不要输出数组、范围、边界框或多个候选坐标。"
))

def _take_context_screenshot(thread_id: str) -> str:
    return take_screenshot.invoke({}, config={"configurable": {"thread_id": thread_id}})


_context_builder = ContextBuilder(_system, screenshot_provider=_take_context_screenshot)
_llm_logger = LLMLogger(log_dir="logs")
_tool_node = ToolNode(TOOLS)


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def agent_node(state: State, config=None) -> dict:
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
    context, screenshot_b64 = _context_builder.build_with_metadata(state["messages"], thread_id=thread_id)
    response = _llm.invoke(context)
    _llm_logger.log(context, response, thread_id=thread_id)
    update = {"messages": [response]}
    if screenshot_b64:
        update["latest_screenshot"] = screenshot_b64
    return update


def _patch_tool_call_args(state: State) -> State:
    from langchain_core.messages import AIMessage
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], AIMessage):
        return state
    last: AIMessage = messages[-1]
    if not last.tool_calls:
        return state
    patched = [
        {**tc, "args": normalize_tool_args(tc["name"], tc.get("args", {}))}
        for tc in last.tool_calls
    ]
    return {**state, "messages": [*messages[:-1], last.model_copy(update={"tool_calls": patched})]}


def tools_node(state: State, config=None) -> dict:
    state = _patch_tool_call_args(state)
    update = _tool_node.invoke(state, config=config)
    update["tool_rounds"] = state.get("tool_rounds", 0) + 1
    if update["tool_rounds"] < MAX_REACT_TOOL_ROUNDS:
        time.sleep(POST_TOOL_SCREEN_SETTLE_SECONDS)
    return update


def after_tools(state: State) -> str:
    if state.get("tool_rounds", 0) >= MAX_REACT_TOOL_ROUNDS:
        return END
    return "agent"


# ── Build Graph ───────────────────────────────────────────────────────────────

_checkpointer = MemorySaver()


def _build_graph():
    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_conditional_edges("tools", after_tools, {"agent": "agent", END: END})
    return builder.compile(checkpointer=_checkpointer)


_graph = _build_graph()

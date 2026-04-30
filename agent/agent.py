import os
import re
import time
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import NotRequired, TypedDict

from agent.context import ContextBuilder
from agent.limits import MAX_REACT_ROUNDS, POST_TOOL_SCREEN_SETTLE_SECONDS
from agent.logger import LLMLogger
from agent.tool_args import normalize_tool_args
from agent.tools import TOOLS, take_screenshot
from llm.provider_config import resolve_chat_provider_config

load_dotenv()

# ── LangGraph State ───────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    latest_screenshot: NotRequired[str]
    pre_tool_screenshot: NotRequired[str]
    tool_rounds: NotRequired[int]
    plan: NotRequired[str]
    plan_steps: NotRequired[list[str]]
    current_step: NotRequired[int]
    complete: NotRequired[bool]
    step_advanced: NotRequired[bool]


# ── LLM + context + logger ────────────────────────────────────────────────────

def _resolve_node_config(prefix: str):
    """Resolve per-node LLM config with global provider settings as fallback."""
    return resolve_chat_provider_config(
        provider=os.getenv(f"{prefix}_API_PROVIDER"),
        model=os.getenv(f"{prefix}_MODEL"),
        api_key=os.getenv(f"{prefix}_API_KEY"),
        base_url=os.getenv(f"{prefix}_BASE_URL"),
    )


_execute_cfg = _resolve_node_config("EXECUTE")
_plan_cfg = _resolve_node_config("PLAN")
_check_cfg = _resolve_node_config("CHECK")

_llm = ChatOpenAI(
    model=_execute_cfg.model,
    api_key=_execute_cfg.api_key,
    base_url=_execute_cfg.base_url,
    extra_body={"enable_thinking": False},
).bind_tools(TOOLS, strict=True, parallel_tool_calls=False)

_plan_llm = ChatOpenAI(
    model=_plan_cfg.model,
    api_key=_plan_cfg.api_key,
    base_url=_plan_cfg.base_url,
    extra_body={"enable_thinking": False},
)

_check_llm = ChatOpenAI(
    model=_check_cfg.model,
    api_key=_check_cfg.api_key,
    base_url=_check_cfg.base_url,
    extra_body={"enable_thinking": False},
)

_system = SystemMessage(content=(
    "你是一个 iPhone 操作助手。每轮用户指令都会附带当前屏幕截图，"
    "你必须根据截图内容和用户指令执行操作。"
    "如果用户请求需要点击、输入、返回主页或打开页面，你不能只输出文字，"
    "必须调用一个合适的工具。"
    "每次调用工具前，可以先用一两句话输出简短可见操作依据："
    "你观察到了什么目标、为什么选择这个动作；随后必须立即调用工具。"
    "只有当任务已经完成、无需任何手机操作时，才可以不调用工具。"
    "如果当前屏幕已经满足用户请求或当前计划步骤，必须直接回复完成，不要调用任何工具。"
    "不要把 go_to_home_screen 当作完成任务后的收尾动作；只有用户要求回到主页或必须先回主页才能继续时才调用它。"
    "输入文字时优先使用 tap_and_type；当前输入工具默认会在输入后按一次回车来提交。"
    "坐标系：左上角(0,0)，右下角(1000,1000)。"
    "调用 tap_screen 时，x 和 y 必须是单个数字，表示点击点中心；"
    "不要输出数组、范围、边界框或多个候选坐标。"
))

_plan_system = SystemMessage(content=(
    "你是一个 iPhone 操作规划助手。根据用户的请求和当前屏幕截图，"
    "列出完成任务需要的操作步骤。\n"
    "格式：\n1. 步骤一\n2. 步骤二\n...\n"
    "只输出编号步骤，不要解释，不要调用工具。"
))

_check_system = SystemMessage(content=(
    "你是一个 iPhone 操作验证助手。\n"
    "你会收到：操作步骤描述、执行前截图、执行后截图。\n\n"
    "请按以下格式分析，每项单独一行：\n"
    "屏幕变化：<执行前后屏幕的主要差异>\n"
    "成功标准：<该步骤完成时屏幕应呈现的状态>\n"
    "结论：YES 或 NO\n\n"
    "判断原则：\n"
    "- YES：屏幕变化与成功标准明确吻合\n"
    "- NO：屏幕无明显变化、与成功标准不符、或仍处于加载中\n"
    "- 不确定时判 NO\n"
    "严格按格式输出，不要添加其他内容。"
))

_verify_system = SystemMessage(content=(
    "你是一个 iPhone 操作验证助手。\n"
    "你会收到：操作步骤描述、当前截图。\n\n"
    "请按以下格式分析，每项单独一行：\n"
    "屏幕描述：<当前屏幕显示的主要内容>\n"
    "成功标准：<该步骤完成时屏幕应呈现的状态>\n"
    "结论：YES 或 NO\n\n"
    "判断原则：\n"
    "- YES：当前屏幕已满足该步骤的完成条件，无需再次执行\n"
    "- NO：该步骤尚未完成，需要执行操作\n"
    "- 不确定时判 NO\n"
    "严格按格式输出，不要添加其他内容。"
))


def _take_context_screenshot(thread_id: str) -> str:
    return take_screenshot.invoke({}, config={"configurable": {"thread_id": thread_id}})


_context_builder = ContextBuilder(_system, screenshot_provider=_take_context_screenshot)
_llm_logger = LLMLogger(log_dir="logs")
_tool_node = ToolNode(TOOLS)


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def plan_node(state: State, config=None) -> dict:
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
    messages = state.get("messages", [])

    # Extract text from the latest human message
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    user_text = ""
    if last_human:
        content = last_human.content
        if isinstance(content, str):
            user_text = content
        elif isinstance(content, list):
            user_text = " ".join(
                b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
            )

    # Take screenshot for planning
    screenshot_b64 = _take_context_screenshot(thread_id) if thread_id else None

    user_content: list = [{"type": "text", "text": user_text}]
    if screenshot_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        })

    context = [_plan_system, HumanMessage(content=user_content)]
    start = time.perf_counter()
    response = _plan_llm.invoke(context)
    duration_s = time.perf_counter() - start
    _llm_logger.log(
        context,
        response,
        thread_id=thread_id,
        node="plan",
        duration_s=duration_s,
        provider=_plan_cfg.provider,
        model=_plan_cfg.model,
    )
    plan_text = response.content if isinstance(response.content, str) else str(response.content)

    plan_steps = re.findall(r"^\d+[.、]\s*(.+)$", plan_text, re.MULTILINE)

    update: dict = {
        "plan": plan_text,
        "plan_steps": plan_steps,
        "current_step": 0,
        "tool_rounds": 0,
    }
    if screenshot_b64:
        update["latest_screenshot"] = screenshot_b64
    return update


def agent_node(state: State, config=None) -> dict:
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
    context, screenshot_b64 = _context_builder.build_with_metadata(state["messages"], thread_id=thread_id)

    # Inject plan with step status into system message if available
    plan_steps = state.get("plan_steps")
    if plan_steps:
        current_step = state.get("current_step", 0)
        annotated = []
        for i, step in enumerate(plan_steps):
            if i < current_step:
                annotated.append(f"✓ {i + 1}. {step}")
            elif i == current_step:
                annotated.append(f"→ {i + 1}. {step}")
            else:
                annotated.append(f"○ {i + 1}. {step}")
        context[0] = SystemMessage(content=context[0].content + "\n\n[本次任务执行计划]\n" + "\n".join(annotated))
    elif state.get("plan"):
        context[0] = SystemMessage(content=context[0].content + f"\n\n[本次任务执行计划]\n{state['plan']}")

    start = time.perf_counter()
    response = _llm.invoke(context)
    duration_s = time.perf_counter() - start
    _llm_logger.log(
        context,
        response,
        thread_id=thread_id,
        node="execute",
        duration_s=duration_s,
        provider=_execute_cfg.provider,
        model=_execute_cfg.model,
    )
    update = {"messages": [response]}
    if screenshot_b64:
        update["latest_screenshot"] = screenshot_b64
    return update


def _patch_tool_call_args(state: State) -> State:
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
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
    pre_screenshot = _take_context_screenshot(thread_id) if thread_id else state.get("latest_screenshot")
    state = _patch_tool_call_args(state)
    update = _tool_node.invoke(state, config=config)
    update["tool_rounds"] = state.get("tool_rounds", 0) + 1
    if pre_screenshot:
        update["pre_tool_screenshot"] = pre_screenshot
    time.sleep(POST_TOOL_SCREEN_SETTLE_SECONDS)
    return update



def check_node(state: State, config=None) -> dict:
    """Check step progress after each tool round; advance current_step based on screenshot."""
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")

    # Extract original user request
    messages = state.get("messages", [])
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    user_text = ""
    if last_human:
        content = last_human.content
        if isinstance(content, str):
            user_text = content
        elif isinstance(content, list):
            user_text = " ".join(
                b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
            )

    plan_steps = state.get("plan_steps", [])
    current_step = state.get("current_step", 0)

    # Take fresh screenshot
    screenshot_b64 = _take_context_screenshot(thread_id) if thread_id else state.get("latest_screenshot")

    # Focus only on the current step
    step_text = plan_steps[current_step] if current_step < len(plan_steps) else ""
    pre_screenshot = state.get("pre_tool_screenshot")
    check_content: list = [{"type": "text", "text": (
        f"任务：{user_text}\n"
        f"当前步骤（第{current_step + 1}步）：{step_text}\n"
    )}]
    if pre_screenshot:
        check_content.append({"type": "text", "text": "【执行前截图】"})
        check_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{pre_screenshot}"},
        })
        check_content.append({"type": "text", "text": "【执行后截图】"})
    if screenshot_b64:
        check_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        })

    context = [_check_system, HumanMessage(content=check_content)]
    start = time.perf_counter()
    response = _check_llm.invoke(context)
    duration_s = time.perf_counter() - start
    _llm_logger.log(
        context,
        response,
        thread_id=thread_id,
        node="check",
        duration_s=duration_s,
        provider=_check_cfg.provider,
        model=_check_cfg.model,
    )

    lines = [l.strip() for l in (response.content or "").strip().splitlines() if l.strip()]
    step_done = False
    for line in reversed(lines):
        upper = line.upper()
        if "YES" in upper or "NO" in upper:
            step_done = "YES" in upper
            break

    new_step = min(current_step + 1, len(plan_steps)) if step_done else current_step
    complete = step_done and new_step >= len(plan_steps)

    update: dict = {"current_step": new_step, "complete": complete}
    if screenshot_b64:
        update["latest_screenshot"] = screenshot_b64
    return update


def verify_node(state: State, config=None) -> dict:
    """Check if current step is already done when agent didn't call any tool."""
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")

    messages = state.get("messages", [])
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    user_text = ""
    if last_human:
        content = last_human.content
        if isinstance(content, str):
            user_text = content
        elif isinstance(content, list):
            user_text = " ".join(
                b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
            )

    plan_steps = state.get("plan_steps", [])
    current_step = state.get("current_step", 0)
    step_text = plan_steps[current_step] if current_step < len(plan_steps) else ""

    screenshot_b64 = _take_context_screenshot(thread_id) if thread_id else state.get("latest_screenshot")

    verify_content: list = [{"type": "text", "text": (
        f"任务：{user_text}\n"
        f"当前步骤（第{current_step + 1}步）：{step_text}\n"
    )}]
    if screenshot_b64:
        verify_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        })

    context = [_verify_system, HumanMessage(content=verify_content)]
    start = time.perf_counter()
    response = _check_llm.invoke(context)
    duration_s = time.perf_counter() - start
    _llm_logger.log(
        context,
        response,
        thread_id=thread_id,
        node="verify",
        duration_s=duration_s,
        provider=_check_cfg.provider,
        model=_check_cfg.model,
    )

    lines = [l.strip() for l in (response.content or "").strip().splitlines() if l.strip()]
    step_done = False
    for line in reversed(lines):
        upper = line.upper()
        if "YES" in upper or "NO" in upper:
            step_done = "YES" in upper
            break

    new_step = min(current_step + 1, len(plan_steps)) if step_done else current_step
    complete = step_done and new_step >= len(plan_steps)

    update: dict = {"current_step": new_step, "complete": complete, "step_advanced": step_done}
    if screenshot_b64:
        update["latest_screenshot"] = screenshot_b64
    if not step_done:
        # Count this no-tool round against the loop budget
        update["tool_rounds"] = state.get("tool_rounds", 0) + 1
        update["messages"] = [HumanMessage(content=(
            f"当前步骤尚未完成，请调用合适的工具来执行：{step_text}"
        ))]
    return update


def after_verify(state: State) -> str:
    if state.get("complete", False):
        return END
    if state.get("step_advanced", False):
        return "agent"
    if state.get("tool_rounds", 0) >= MAX_REACT_ROUNDS:
        return END
    if not state.get("plan_steps"):
        return END
    return "agent"


def after_agent(state: State) -> str:
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage):
        last = messages[-1]
        content = last.content if isinstance(last.content, str) else ""
        done_markers = ("任务已完成", "已经完成", "无需进一步操作", "无需操作")
        if last.tool_calls and any(marker in content for marker in done_markers):
            return END
        if last.tool_calls:
            return "tools"
    if state.get("complete", False):
        return END
    if not state.get("plan_steps"):
        return END
    if state.get("tool_rounds", 0) >= MAX_REACT_ROUNDS:
        return END
    return "verify"


def after_check(state: State) -> str:
    if state.get("complete", False):
        return END
    if state.get("tool_rounds", 0) >= MAX_REACT_ROUNDS:
        return END
    if not state.get("plan_steps"):
        return END
    return "agent"


# ── Build Graph ───────────────────────────────────────────────────────────────

_checkpointer = MemorySaver()


def _build_graph():
    builder = StateGraph(State)
    builder.add_node("plan", plan_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("verify", verify_node)
    builder.add_node("check", check_node)
    builder.set_entry_point("plan")
    builder.add_edge("plan", "agent")
    builder.add_conditional_edges("agent", after_agent, {"tools": "tools", "verify": "verify", END: END})
    builder.add_edge("tools", "check")
    builder.add_conditional_edges("verify", after_verify, {"agent": "agent", END: END})
    builder.add_conditional_edges("check", after_check, {"agent": "agent", END: END})
    return builder.compile(checkpointer=_checkpointer)


_graph = _build_graph()

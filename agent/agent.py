import os
import re
import time
import uuid
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
from agent.utils import ocr_from_b64
from llm.provider_config import resolve_chat_provider_config

load_dotenv()

# ── LangGraph State ───────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    latest_screenshot: NotRequired[str]
    pre_tool_screenshot: NotRequired[str]
    tool_rounds: NotRequired[int]
    last_actions: NotRequired[list[dict]]
    react_trace: NotRequired[list[dict]]
    complete: NotRequired[bool]
    recovered_tool: NotRequired[bool]
    recover_injected: NotRequired[bool]
    logical_verify_decision: NotRequired[str]
    final_result: NotRequired[str]


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
_check_cfg = _resolve_node_config("CHECK")
_output_cfg = _resolve_node_config("OUTPUT")

_llm = ChatOpenAI(
    model=_execute_cfg.model,
    api_key=_execute_cfg.api_key,
    base_url=_execute_cfg.base_url,
    extra_body={"enable_thinking": False},
).bind_tools(TOOLS, strict=True, parallel_tool_calls=False)

_check_llm = ChatOpenAI(
    model=_check_cfg.model,
    api_key=_check_cfg.api_key,
    base_url=_check_cfg.base_url,
    extra_body={"enable_thinking": False},
)

_output_llm = ChatOpenAI(
    model=_output_cfg.model,
    api_key=_output_cfg.api_key,
    base_url=_output_cfg.base_url,
    extra_body={"enable_thinking": False},
)

_system = SystemMessage(content=(
    "你是一个 iPhone 操作助手。每轮用户指令都会附带当前屏幕截图，"
    "你必须根据截图内容和用户指令执行操作。"
    "不要假设应用处在主界面；手机可能停留在任意中间页面、弹窗、详情页、搜索页或上一次任务留下的状态。"
    "每一轮都必须先观察当前截图，再决定下一步最小必要动作。"
    "如果当前用户请求需要点击、输入、滑动、返回主页、打开页面或切换界面，"
    "你不能只输出文字，必须调用一个合适的工具。"
    "每次调用工具前，可以先用一两句话输出简短可见操作依据："
    "你观察到了什么目标、为什么选择这个动作；随后必须立即调用工具。"
    "只有当当前屏幕已经明确满足用户最终请求、无需任何手机操作时，才可以不调用工具并回复任务已完成。"
    "判断任务是否完成时，必须看页面主内容或明确结果，不要只看目标控件是否可见、位于指定位置、或看起来被选中。"
    "如果要进入某个入口，当前屏幕只是显示入口本身不算完成，仍需点击进入下一层页面。"
    "不要把 go_to_home_screen 当作完成任务后的收尾动作；只有用户要求回到主页或必须先回主页才能继续时才调用它。"
    "输入文字时优先使用 tap_and_type；当前输入工具默认会在输入后按一次回车来提交。"
    "坐标系：左上角(0,0)，右下角(1000,1000)。"
    "调用 tap_screen 时，x 和 y 必须是单个数字，表示点击点中心；"
    "不要输出数组、范围、边界框或多个候选坐标。"
    "禁止调用 tap_screen({}) 或 tap_and_type({}) 这类空参数工具；"
    "如果要点击，必须先从截图估计一个明确的中心点坐标并传入 x、y。"
))

_check_system = SystemMessage(content=(
    "你是一个 iPhone 操作验证助手。\n"
    "你会收到：用户任务、执行前截图、执行后截图。\n\n"
    "请按以下格式分析，每项单独一行：\n"
    "屏幕变化：<执行前后屏幕的主要差异>\n"
    "成功标准：<用户任务完成时屏幕应呈现的最终状态或明确结果>\n"
    "结论：YES 或 NO\n\n"
    "判断原则：\n"
    "- YES：执行后屏幕已经明确满足用户任务的最终需求，无需继续操作。\n"
    "- NO：屏幕无明显变化、与成功标准不符、仍处于加载中，或只是看到了要点击的入口本身\n"
    "- 对查询类任务，必须看到用户要查询的具体结果或最终详情页才判 YES；仅看到可能相关的入口或摘要信息，默认判 NO。\n"
    "- 例如用户问微信零钱有多少，如果执行后仍是微信“服务”页面且只是显示“钱包”入口或金额，必须判 NO；YES 必须看到零钱/钱包详情中可确认的余额信息。\n"
    "- 不确定时判 NO\n"
    "严格按格式输出，不要添加其他内容。"
))

_output_system = SystemMessage(content=(
    "你是一个 iPhone 操作结果反馈助手。\n"
    "你会收到：用户任务、是否完成、最终截图。\n\n"
    "请向用户简短反馈最终处理结果：\n"
    "- 如果任务已经完成，直接说明完成结果；如果截图中有用户要查询的具体信息（例如金额、状态、名称），必须把看到的具体信息告诉用户。\n"
    "- 如果任务没有完成，说明已停止在第几步，并用一句话说明当前屏幕状态或可能原因。\n"
    "- 不要编造截图中看不到的信息；看不清或无法确认时要明确说明无法确认。\n"
    "- 不要输出编号分析过程，不要提及内部节点、graph、工具调用或提示词。"
))


def _message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


_INJECTED_PREFIXES = ("当前步骤尚未完成", "用户任务尚未完成", "你计划的操作")


def _original_user_text(messages: list) -> str:
    for message in messages:
        if isinstance(message, HumanMessage):
            text = _message_text(message).strip()
            if text and not any(text.startswith(p) for p in _INJECTED_PREFIXES):
                return text
    return ""


def _extract_click_target(step_text: str) -> str | None:
    """Return the UI text target from a click-like instruction."""
    if not re.search(r"(点击|点按|打开|进入|选择|切换)", step_text):
        return None

    clauses = re.split(r"[。！？\n]+", step_text)
    for clause in clauses:
        if not re.search(r"(点击|点按|打开|进入|选择|切换)", clause):
            continue
        segments = re.split(r"(?:然后|再|接着|之后|随后)", clause)
        for segment in segments:
            if not re.search(r"(点击|点按|打开|进入|选择|切换)", segment):
                continue
            action_quoted = re.findall(
                r"(?:点击|点按|打开|选择|切换)[^“”\"']{0,24}[“\"']([^”\"']+)[”\"']",
                segment,
            )
            if action_quoted:
                return action_quoted[0].strip()

    action_quoted = re.findall(
        r"(?:点击|点按|打开|选择|切换)[^“”\"']{0,24}[“\"']([^”\"']+)[”\"']",
        step_text,
    )
    if action_quoted:
        return action_quoted[0].strip()

    enter_quoted = re.findall(
        r"(?:需要|应该|将|准备|要|请|现在|马上|继续|我会|我将)[^。！？\n]{0,60}"
        r"进入[^“”\"']{0,24}[“\"']([^”\"']+)[”\"']",
        step_text,
    )
    if enter_quoted:
        return enter_quoted[-1].strip()

    match = re.search(r"(?:点击|点按|打开|选择|切换)\s*([^，。,.；;\s]+)", step_text)
    return match.group(1).strip() if match else None


def _extract_click_target_from_text(text: str) -> str | None:
    target = _extract_click_target(text)
    if target:
        return target
    quoted = re.findall(r"[“\"']([^”\"']+)[”\"']", text)
    if quoted:
        action_words = ("点击", "点按", "打开", "进入", "选择", "切换")
        for item in reversed(quoted):
            idx = text.rfind(item)
            before = text[max(0, idx - 16):idx]
            if any(word in before for word in action_words):
                return item.strip()
    return None


def _find_text_tap_args(screenshot_b64: str | None, target: str | None) -> dict | None:
    """Locate target text in the screenshot and return normalized tap args."""
    if not screenshot_b64 or not target:
        return None
    try:
        results, (width, height) = ocr_from_b64(screenshot_b64)
    except Exception:
        return None

    candidates = [
        result for result in results
        if target in result.text or result.text in target
    ]
    if not candidates:
        return None

    exact = [result for result in candidates if result.text == target]
    target_result = max(exact or candidates, key=lambda result: result.confidence)
    px, py = target_result.tap_coords(width, height)
    return {
        "x": max(0, min(1000, px / width * 1000)),
        "y": max(0, min(1000, py / height * 1000)),
    }


def _tap_call_from_last_response(state: State) -> AIMessage | None:
    messages = state.get("messages", [])
    step_text = ""
    if messages and isinstance(messages[-1], AIMessage):
        step_text = _message_text(messages[-1])
    target = _extract_click_target(step_text)
    if not target:
        target = _extract_click_target_from_text(step_text)
    args = _find_text_tap_args(state.get("latest_screenshot"), target)
    if not args:
        return None
    return AIMessage(
        content=f"根据 OCR 定位到“{target}”，现在点击该入口。",
        tool_calls=[{
            "name": "tap_screen",
            "args": args,
            "id": f"ocr_tap_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        }],
    )


def _completion_like_text(text: str) -> bool:
    return any(marker in text for marker in ("任务已完成", "已经完成", "无需进一步操作", "无需操作"))


def _format_action_args(args: dict | None) -> str:
    if not args:
        return "{}"
    parts = []
    for key, value in args.items():
        if isinstance(value, float):
            value = round(value, 1)
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _format_react_trace(trace: list[dict], limit: int = 5) -> str:
    recent = trace[-limit:]
    lines = []
    for item in recent:
        actions = item.get("actions") or []
        action_text = "; ".join(
            f"{action.get('name', 'unknown')}({_format_action_args(action.get('args'))})"
            for action in actions
        ) or "无工具动作"
        result = item.get("result", "UNKNOWN")
        summary = item.get("summary", "")
        lines.append(f"{item.get('round', '?')}. {action_text} -> {result} {summary}".strip())
    return "\n".join(lines)


def _parse_check_result(content: str) -> tuple[bool, str]:
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    step_done = False
    for line in reversed(lines):
        upper = line.upper()
        if "YES" in upper or "NO" in upper:
            step_done = "YES" in upper
            break
    summary = ""
    for line in lines:
        if line.startswith(("屏幕变化：", "屏幕描述：")):
            summary = line.split("：", 1)[-1].strip()
            break
    return step_done, summary


def _get_tap_xy(args: dict) -> tuple[float | None, float | None]:
    """Extract (x, y) from tap args, handling malformed array inputs."""
    x = args.get("x")
    y = args.get("y")
    if isinstance(x, (list, tuple)) and len(x) >= 2 and y is None:
        x, y = x[0], x[1]
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None, None


def _is_repeated_failed_tap(state: State) -> bool:
    """Return True if the last planned tap matches a recently failed tap location."""
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], AIMessage):
        return False
    last = messages[-1]
    if not last.tool_calls:
        return False
    trace = state.get("react_trace", [])
    failed_taps = [
        action
        for entry in trace
        if entry.get("result") == "NO"
        for action in entry.get("actions", [])
        if action.get("name") in ("tap_screen", "tap_and_type")
    ]
    if not failed_taps:
        return False
    for tc in last.tool_calls:
        if tc.get("name") not in ("tap_screen", "tap_and_type"):
            continue
        tx, ty = _get_tap_xy(tc.get("args", {}))
        if tx is None:
            continue
        for fa in failed_taps[-3:]:
            fx, fy = _get_tap_xy(fa.get("args", {}))
            if fx is None:
                continue
            if abs(tx - fx) < 50 and abs(ty - fy) < 50:
                return True
    return False


def _logical_verify_decision(user_text: str, screenshot_b64: str | None) -> tuple[str, str]:
    """Return complete/incomplete/unknown from cheap deterministic screen rules."""
    if not screenshot_b64:
        return "unknown", "no screenshot available"
    try:
        results, _ = ocr_from_b64(screenshot_b64)
    except Exception as exc:
        return "unknown", f"OCR failed: {exc}"

    screen_text = "\n".join(result.text for result in results)
    query_like = any(word in user_text for word in ("多少", "余额", "零钱", "还有多少钱"))
    asks_wechat_balance = "微信" in user_text and ("零钱" in user_text or "钱" in user_text)

    if asks_wechat_balance:
        has_intermediate_wallet_entry = "服务" in screen_text and "钱包" in screen_text
        has_balance_detail = "零钱" in screen_text and any(word in screen_text for word in ("余额", "可用", "明细"))
        if has_intermediate_wallet_entry and not has_balance_detail:
            return "incomplete", "当前仍像微信服务页/钱包入口页，只看到入口或摘要，尚未进入可确认微信零钱的详情页"

    if query_like and any(word in screen_text for word in ("加载中", "正在加载", "网络")):
        return "incomplete", "当前页面仍在加载或网络状态不稳定"

    return "unknown", "no deterministic rule matched"


def _take_context_screenshot(thread_id: str) -> str:
    return take_screenshot.invoke({}, config={"configurable": {"thread_id": thread_id}})


_context_builder = ContextBuilder(_system, screenshot_provider=_take_context_screenshot)
_llm_logger = LLMLogger(log_dir="logs")
_tool_node = ToolNode(TOOLS)


# ── Graph Nodes ───────────────────────────────────────────────────────────────


def agent_node(state: State, config=None) -> dict:
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
    context, screenshot_b64 = _context_builder.build_with_metadata(state["messages"], thread_id=thread_id)
    react_trace = state.get("react_trace", [])
    if react_trace:
        context[0] = SystemMessage(content=(
            context[0].content
            + "\n\n[本轮已尝试操作]\n"
            + _format_react_trace(react_trace)
            + "\n\n如果同一动作或同一坐标已经失败，不要机械重复；请换一个更精确的位置、返回/滚动/搜索，或在无法继续时说明原因。"
        ))

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
    step_text = _message_text(last)
    target = _extract_click_target(step_text)
    if not target:
        target = _extract_click_target_from_text(step_text)
    patched = []
    for tc in last.tool_calls:
        args = normalize_tool_args(tc["name"], tc.get("args", {}))
        if tc["name"] in ("tap_screen", "tap_and_type") and ("x" not in args or "y" not in args):
            ocr_args = _find_text_tap_args(state.get("latest_screenshot"), target)
            if ocr_args:
                args = {**args, **ocr_args}
        patched.append({**tc, "args": args})
    return {**state, "messages": [*messages[:-1], last.model_copy(update={"tool_calls": patched})]}


def tools_node(state: State, config=None) -> dict:
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
    pre_screenshot = _take_context_screenshot(thread_id) if thread_id else state.get("latest_screenshot")
    state = _patch_tool_call_args(state)
    messages = state.get("messages", [])
    last_actions = []
    if messages and isinstance(messages[-1], AIMessage):
        for tool_call in messages[-1].tool_calls:
            last_actions.append({
                "name": tool_call.get("name", "unknown"),
                "args": tool_call.get("args", {}),
            })
    update = _tool_node.invoke(state, config=config)
    update["tool_rounds"] = state.get("tool_rounds", 0) + 1
    update["last_actions"] = last_actions
    update["recovered_tool"] = False
    if pre_screenshot:
        update["pre_tool_screenshot"] = pre_screenshot
    time.sleep(POST_TOOL_SCREEN_SETTLE_SECONDS)
    return update



def recover_node(state: State, config=None) -> dict:
    """Recover missing tool calls using deterministic rules before falling back to verify."""
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], AIMessage):
        _llm_logger.log_logical(
            thread_id=thread_id,
            node="recover",
            decision="skip",
            reason="last message is not an AIMessage",
        )
        return {"recovered_tool": False}

    last = messages[-1]
    content = last.content if isinstance(last.content, str) else ""

    if last.tool_calls and _completion_like_text(content):
        _llm_logger.log_logical(
            thread_id=thread_id,
            node="recover",
            content=content,
            decision="skip",
            reason="completion-like content with tool calls",
        )
        return {"recovered_tool": False, "recover_injected": False}

    if last.tool_calls:
        # Routed here because after_agent detected a repeated failed action
        user_text = _original_user_text(messages)
        trace = state.get("react_trace", [])
        failed_desc = "; ".join(
            f"{a.get('name')}({_format_action_args(a.get('args'))})"
            for entry in trace[-3:]
            if entry.get("result") == "NO"
            for a in entry.get("actions", [])
        ) or "之前的操作"
        trace = [
            *trace,
            {
                "round": state.get("tool_rounds", 0) + 1,
                "actions": [{"name": tc["name"], "args": tc.get("args", {})} for tc in last.tool_calls],
                "result": "NO",
                "summary": "重复了已失败的点击位置，已阻止执行",
            },
        ][-8:]
        _llm_logger.log_logical(
            thread_id=thread_id,
            node="recover",
            content=content,
            decision="blocked_repeat",
            reason=f"repeated failed tap detected: {failed_desc}",
        )
        return {
            "messages": [HumanMessage(content=(
                f"你计划的操作（{failed_desc}）与之前已失败的点击位置相同，已被阻止。"
                f"请仔细观察截图，换一个完全不同的方式继续：{user_text}"
            ))],
            "tool_rounds": state.get("tool_rounds", 0) + 1,
            "react_trace": trace,
            "recovered_tool": False,
            "recover_injected": True,
        }

    fallback_call = _tap_call_from_last_response(state)
    if fallback_call:
        tool_call = fallback_call.tool_calls[0] if fallback_call.tool_calls else None
        _llm_logger.log_logical(
            thread_id=thread_id,
            node="recover",
            content=content,
            decision="recover_tool",
            reason="model described an action but emitted no structured tool call; OCR located a target",
            tool_call=tool_call,
        )
        return {"messages": [fallback_call], "recovered_tool": True, "recover_injected": False}

    trace = state.get("react_trace", [])
    trace = [
        *trace,
        {
            "round": state.get("tool_rounds", 0) + 1,
            "actions": [],
            "result": "NO",
            "summary": "模型未调用工具，规则未能补出可靠工具调用",
        },
    ][-8:]
    _llm_logger.log_logical(
        thread_id=thread_id,
        node="recover",
        content=content,
        decision="fallback_verify",
        reason="no reliable tool call could be recovered by rules",
    )
    return {"react_trace": trace, "recovered_tool": False, "recover_injected": False}


def check_node(state: State, config=None) -> dict:
    """Check task progress after each tool round based on the current screenshot."""
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")

    messages = state.get("messages", [])
    user_text = _original_user_text(messages)

    # Take fresh screenshot
    screenshot_b64 = _take_context_screenshot(thread_id) if thread_id else state.get("latest_screenshot")

    pre_screenshot = state.get("pre_tool_screenshot")
    check_content: list = [{"type": "text", "text": (
        f"任务：{user_text}\n"
        "请判断这次工具执行后，用户任务是否已经完成。\n"
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

    check_text = response.content if isinstance(response.content, str) else str(response.content or "")
    step_done, summary = _parse_check_result(check_text)
    complete = step_done

    trace = state.get("react_trace", [])
    actions = state.get("last_actions", [])
    if actions:
        trace = [
            *trace,
            {
                "round": state.get("tool_rounds", 0),
                "actions": actions,
                "result": "YES" if step_done else "NO",
                "summary": summary,
            },
        ][-8:]

    update: dict = {"complete": complete, "react_trace": trace, "last_actions": [], "recovered_tool": False}
    if screenshot_b64:
        update["latest_screenshot"] = screenshot_b64
    return update


def logical_verify_node(state: State, config=None) -> dict:
    """Apply deterministic completion rules before spending an LLM verify call."""
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
    messages = state.get("messages", [])
    user_text = _original_user_text(messages)
    screenshot_b64 = _take_context_screenshot(thread_id) if thread_id else state.get("latest_screenshot")
    decision, reason = _logical_verify_decision(user_text, screenshot_b64)

    _llm_logger.log_logical(
        thread_id=thread_id,
        node="logical_verify",
        content=user_text,
        decision=decision,
        reason=reason,
    )

    update: dict = {"logical_verify_decision": decision, "recovered_tool": False}
    if screenshot_b64:
        update["latest_screenshot"] = screenshot_b64

    if decision == "complete":
        update["complete"] = True
    elif decision in ("incomplete", "unknown"):
        trace = state.get("react_trace", [])
        trace = [
            *trace,
            {
                "round": state.get("tool_rounds", 0) + 1,
                "actions": [],
                "result": "NO",
                "summary": reason,
            },
        ][-8:]
        update["tool_rounds"] = state.get("tool_rounds", 0) + 1
        update["react_trace"] = trace
        update["complete"] = False
        update["messages"] = [HumanMessage(content=(
            f"用户任务尚未完成，请根据当前屏幕继续操作：{user_text}"
        ))]
    return update


def output_node(state: State, config=None) -> dict:
    """Generate the final user-facing result for this task."""
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
    messages = state.get("messages", [])
    user_text = _original_user_text(messages)

    complete = state.get("complete", False)
    screenshot_b64 = _take_context_screenshot(thread_id) if thread_id else state.get("latest_screenshot")

    output_content: list = [{"type": "text", "text": (
        f"用户任务：{user_text}\n"
        f"任务是否完成：{'是' if complete else '否'}\n"
        "请根据以上信息和最终截图，给用户反馈最终处理结果。"
    )}]
    if screenshot_b64:
        output_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        })

    context = [_output_system, HumanMessage(content=output_content)]
    start = time.perf_counter()
    response = _output_llm.invoke(context)
    duration_s = time.perf_counter() - start
    _llm_logger.log(
        context,
        response,
        thread_id=thread_id,
        node="output",
        duration_s=duration_s,
        provider=_output_cfg.provider,
        model=_output_cfg.model,
    )

    final_result = response.content if isinstance(response.content, str) else str(response.content)
    update: dict = {"messages": [AIMessage(content=final_result)], "final_result": final_result}
    if screenshot_b64:
        update["latest_screenshot"] = screenshot_b64
    return update


def after_recover(state: State) -> str:
    if state.get("recovered_tool", False):
        return "tools"
    if state.get("recover_injected", False):
        return "agent"
    return "logical_verify"


def after_logical_verify(state: State) -> str:
    if state.get("complete", False):
        return "output"
    if state.get("tool_rounds", 0) >= MAX_REACT_ROUNDS:
        return "output"
    return "agent"


def after_agent(state: State) -> str:
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage):
        last = messages[-1]
        content = last.content if isinstance(last.content, str) else ""
        done_markers = ("任务已完成", "已经完成", "无需进一步操作", "无需操作")
        if last.tool_calls and any(marker in content for marker in done_markers):
            return "output"
        if last.tool_calls:
            if _is_repeated_failed_tap(state):
                return "recover"
            return "tools"
    if state.get("complete", False):
        return "output"
    if state.get("tool_rounds", 0) >= MAX_REACT_ROUNDS:
        return "output"
    return "recover"


def after_check(state: State) -> str:
    if state.get("complete", False):
        return "output"
    if state.get("tool_rounds", 0) >= MAX_REACT_ROUNDS:
        return "output"
    return "agent"


# ── Build Graph ───────────────────────────────────────────────────────────────

_checkpointer = MemorySaver()


def _build_graph():
    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("recover", recover_node)
    builder.add_node("logical_verify", logical_verify_node)
    builder.add_node("check", check_node)
    builder.add_node("output", output_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", after_agent, {"tools": "tools", "recover": "recover", "output": "output"})
    builder.add_conditional_edges("recover", after_recover, {"tools": "tools", "logical_verify": "logical_verify"})
    builder.add_conditional_edges("logical_verify", after_logical_verify, {"agent": "agent", "output": "output"})
    builder.add_edge("tools", "check")
    builder.add_conditional_edges("check", after_check, {"agent": "agent", "output": "output"})
    builder.add_edge("output", END)
    return builder.compile(checkpointer=_checkpointer)


_graph = _build_graph()

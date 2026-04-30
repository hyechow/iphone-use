"""LLM call logger.

Records every LLM call (input context + output) to:
  1. Terminal – rich-formatted, human-readable (great for live debugging)
  2. JSONL file – structured, machine-readable (for offline analysis)

Usage::

    from agent.logger import LLMLogger
    logger = LLMLogger(log_dir="logs")          # or log_dir=None for console-only
    logger.log(context_messages, response, thread_id="session-1")
"""
import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from rich.console import Console
from rich.markup import escape
from rich.rule import Rule
from rich.text import Text

_console = Console(highlight=False)
_TOKEN_ENCODER = None

# ── Message summarisation helpers ─────────────────────────────────────────────

def _summarise_content(content) -> str:
    """Return a short human-readable summary of message content."""
    if isinstance(content, str):
        if len(content) > 200 and _looks_like_base64(content):
            kb = len(content) * 3 // 4 // 1024
            return f"[screenshot ~{kb}kb]"
        return content[:300] + ("…" if len(content) > 300 else "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                t = item["text"]
                parts.append(t[:200] + ("…" if len(t) > 200 else ""))
            elif item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:image"):
                    # data:image/png;base64,<data>
                    data_part = url.split(",", 1)[-1] if "," in url else url
                    kb = len(data_part) * 3 // 4 // 1024
                    parts.append(f"[screenshot ~{kb}kb]")
                else:
                    parts.append(f"[image_url: {url[:60]}]")
        return " | ".join(parts) if parts else "(empty)"
    return repr(content)[:200]


def _looks_like_base64(s: str) -> bool:
    if len(s) < 100:
        return False
    sample = s[:64].rstrip("=")
    import re
    return bool(re.fullmatch(r"[A-Za-z0-9+/]+", sample))


def _msg_to_dict(msg: BaseMessage) -> dict:
    """Serialise a message to a JSON-compatible dict (images truncated)."""
    role = msg.type
    content = msg.content

    if isinstance(content, str):
        if _looks_like_base64(content):
            serialised_content = f"[base64 ~{len(content) * 3 // 4 // 1024}kb]"
        else:
            serialised_content = content
    elif isinstance(content, list):
        serialised_content = []
        for item in content:
            if not isinstance(item, dict):
                serialised_content.append(item)
                continue
            if item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:image"):
                    data_part = url.split(",", 1)[-1] if "," in url else url
                    kb = len(data_part) * 3 // 4 // 1024
                    serialised_content.append({"type": "image_url", "size_kb": kb})
                else:
                    serialised_content.append(item)
            else:
                serialised_content.append(item)
    else:
        serialised_content = repr(content)

    result: dict = {"role": role, "content": serialised_content}

    # AIMessage extras
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = tool_calls

    # ToolMessage extras
    if isinstance(msg, ToolMessage):
        result["tool_name"] = getattr(msg, "name", None)
        result["tool_call_id"] = getattr(msg, "tool_call_id", None)

    usage = _extract_token_usage(msg)
    if usage:
        result["usage"] = usage

    return result


def _raw_msg_to_dict(msg: BaseMessage) -> dict:
    """Return the closest JSON-serialisable representation of a LangChain message."""
    if hasattr(msg, "model_dump"):
        try:
            raw = msg.model_dump(mode="json")
        except TypeError:
            raw = msg.model_dump()
    elif hasattr(msg, "dict"):
        raw = msg.dict()
    else:
        raw = _msg_to_dict(msg)

    raw.pop("invalid_tool_calls", None)

    for attr in (
        "content",
        "tool_calls",
        "additional_kwargs",
        "response_metadata",
        "id",
        "name",
    ):
        value = getattr(msg, attr, None)
        if value not in (None, [], {}):
            raw[attr] = value

    return json.loads(json.dumps(raw, ensure_ascii=False, default=str))


def _normalize_token_usage(usage: dict | None) -> dict:
    if not usage:
        return {}

    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated": False,
        "raw": usage,
    }


def _has_any_tokens(usage: dict) -> bool:
    return any(
        isinstance(usage.get(key), int | float)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    )


def _has_positive_tokens(usage: dict) -> bool:
    return any(
        isinstance(usage.get(key), int | float) and usage[key] > 0
        for key in ("input_tokens", "output_tokens", "total_tokens")
    )


def _extract_token_usage(msg: BaseMessage) -> dict:
    """Return normalized token usage from common LangChain/OpenAI fields."""
    candidates = []

    usage = getattr(msg, "usage_metadata", None)
    if usage:
        candidates.append(_normalize_token_usage(usage))

    metadata = getattr(msg, "response_metadata", {}) or {}
    for key in ("token_usage", "usage"):
        token_usage = metadata.get(key)
        if token_usage:
            candidates.append(_normalize_token_usage(token_usage))

    additional_kwargs = getattr(msg, "additional_kwargs", {}) or {}
    for key in ("token_usage", "usage"):
        token_usage = additional_kwargs.get(key)
        if token_usage:
            candidates.append(_normalize_token_usage(token_usage))

    for candidate in candidates:
        if _has_positive_tokens(candidate):
            return candidate

    return {}


def _message_text_for_estimate(msg: BaseMessage) -> str:
    parts: list[str] = [msg.type]
    content = msg.content
    if isinstance(content, str):
        if not _looks_like_base64(content):
            parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "image_url":
                parts.append("[image]")

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        parts.append(json.dumps(tool_calls, ensure_ascii=False, default=str))

    if isinstance(msg, ToolMessage) and getattr(msg, "name", None):
        parts.append(str(getattr(msg, "name", "")))

    return "\n".join(part for part in parts if part)


def _count_text_tokens(text: str) -> int:
    global _TOKEN_ENCODER
    if not text:
        return 0
    try:
        if _TOKEN_ENCODER is None:
            import tiktoken

            _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        return len(_TOKEN_ENCODER.encode(text))
    except Exception:
        # Rough fallback for mixed Chinese/English when tiktoken is unavailable.
        return max(1, len(text) // 4)


def _estimate_token_usage(context: list[BaseMessage], response: BaseMessage) -> dict:
    input_text = "\n\n".join(_message_text_for_estimate(msg) for msg in context)
    output_text = _message_text_for_estimate(response)
    input_tokens = _count_text_tokens(input_text)
    output_tokens = _count_text_tokens(output_text)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated": True,
        "note": "local text estimate; image token cost is not included",
    }


def _resolve_usage(context: list[BaseMessage], response: BaseMessage) -> dict:
    usage = _extract_token_usage(response)
    if _has_any_tokens(usage):
        return usage
    return _estimate_token_usage(context, response)


# ── LLMLogger ─────────────────────────────────────────────────────────────────

class LLMLogger:
    """Logs every LLM call to the terminal and optionally to a JSONL file."""

    def __init__(self, log_dir: str | Path | None = None):
        self._call_index = 0
        self._logical_index = 0
        self._log_file: Path | None = None
        self._logical_log_file: Path | None = None

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = log_dir / "llm_calls.jsonl"
            self._logical_log_file = log_dir / "logical_calls.jsonl"

    def log(
        self,
        context: list[BaseMessage],
        response: BaseMessage,
        thread_id: str = "",
        node: str = "",
        duration_s: float | None = None,
        provider: str = "",
        model: str = "",
    ) -> None:
        """Log one LLM call. Call this right after llm.invoke()."""
        self._call_index += 1
        ts = datetime.now()

        self._print_to_terminal(
            context,
            response,
            thread_id,
            ts,
            node=node,
            duration_s=duration_s,
            provider=provider,
            model=model,
        )

        if self._log_file is not None:
            self._append_jsonl(
                context,
                response,
                thread_id,
                ts,
                node=node,
                duration_s=duration_s,
                provider=provider,
                model=model,
            )

    def log_logical(
        self,
        *,
        thread_id: str = "",
        node: str = "logical",
        content: str = "",
        decision: str = "",
        reason: str = "",
        tool_call: dict | None = None,
    ) -> None:
        """Log one deterministic logical decision."""
        self._logical_index += 1
        ts = datetime.now()
        ts_str = ts.strftime("%H:%M:%S")
        header = f"Logical Call #{self._logical_index}  [{node}]"
        if thread_id:
            header += f"  thread={thread_id}"
        header += f"  {ts_str}"

        _console.print()
        _console.print(Rule(f"[bold cyan]{header}[/]", style="cyan"))
        _console.print("[bold yellow]▶ INPUT[/]")
        _console.print(f"  content  {escape(content or '(empty)')}")
        _console.print("[bold yellow]◀ OUTPUT[/]")
        _console.print(f"  decision {escape(decision or '(none)')}")
        if reason:
            _console.print(f"  reason   {escape(reason)}")
        if tool_call:
            name = tool_call.get("name", "unknown")
            args = json.dumps(tool_call.get("args", {}), ensure_ascii=False)
            _console.print(f"  [cyan]→ {escape(name)}[/]({escape(args)})")

        if self._logical_log_file is not None:
            record = {
                "index": self._logical_index,
                "timestamp": ts.isoformat(),
                "thread_id": thread_id,
                "node": node,
                "content": content,
                "decision": decision,
                "reason": reason,
                "tool_call": tool_call,
            }
            with self._logical_log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # ── Terminal output ────────────────────────────────────────────────────

    def _print_to_terminal(
        self,
        context: list[BaseMessage],
        response: BaseMessage,
        thread_id: str,
        ts: datetime,
        node: str = "",
        duration_s: float | None = None,
        provider: str = "",
        model: str = "",
    ) -> None:
        idx = self._call_index
        ts_str = ts.strftime("%H:%M:%S")
        header = f"LLM Call #{idx}"
        if node:
            header += f"  {escape(f'[{node}]')}"
        if thread_id:
            header += f"  thread={thread_id}"
        header += f"  {ts_str}"

        _console.print()
        _console.print(Rule(f"[bold cyan]{header}[/]", style="cyan"))
        if provider or model:
            _console.print(f"[dim]provider={escape(provider or '?')}  model={escape(model or '?')}[/]")

        # ── Input ──
        _console.print(f"[bold yellow]▶ INPUT[/] ({len(context)} messages)")
        for i, msg in enumerate(context):
            role_style = {
                "system": "dim white",
                "human": "green",
                "ai": "blue",
                "tool": "magenta",
            }.get(msg.type, "white")

            role_label = f"[{role_style}]{msg.type:8}[/]"
            content = msg.content

            if isinstance(msg, ToolMessage) and getattr(msg, "name", None):
                _console.print(f"  [{i}] {role_label} [bold]{msg.name}[/]")
            else:
                _console.print(f"  [{i}] {role_label}")

            # Print full content
            if isinstance(content, str):
                if _looks_like_base64(content):
                    kb = len(content) * 3 // 4 // 1024
                    _console.print(f"       [dim]screenshot ~{kb}kb[/]")
                else:
                    _console.print(f"       {escape(content)}")
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        _console.print(f"       {escape(item['text'])}")
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            data_part = url.split(",", 1)[-1] if "," in url else url
                            kb = len(data_part) * 3 // 4 // 1024
                            _console.print(f"       [dim]screenshot ~{kb}kb[/]")
                        else:
                            _console.print(f"       [dim]image_url: {escape(url[:80])}[/]")

        # ── Output ──
        _console.print(f"[bold yellow]◀ OUTPUT[/]")

        resp_content = _summarise_content(response.content)
        if resp_content and resp_content != "(empty)":
            _console.print(f"  content  {escape(resp_content)}")

        tool_calls = getattr(response, "tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)
                _console.print(f"  [cyan]→ {tc['name']}[/]({args_str})")

        if not resp_content and not tool_calls:
            _console.print("  (no content, no tool calls)")

        usage = _resolve_usage(context, response)
        if usage:
            prompt = usage.get("input_tokens", "?")
            completion = usage.get("output_tokens", "?")
            total = usage.get("total_tokens", "?")
            suffix = " estimated" if usage.get("estimated") else ""
            _console.print(
                f"  [dim]tokens{suffix}: {prompt} in / {completion} out / {total} total[/]",
                end="\n",
            )
        else:
            _console.print("  [dim]tokens: unavailable[/]")

        if duration_s is not None:
            _console.print(f"  [dim]latency: {duration_s:.2f}s[/]")

        # RAW OUTPUT disabled

    # ── JSONL output ───────────────────────────────────────────────────────

    def _append_jsonl(
        self,
        context: list[BaseMessage],
        response: BaseMessage,
        thread_id: str,
        ts: datetime,
        node: str = "",
        duration_s: float | None = None,
        provider: str = "",
        model: str = "",
    ) -> None:
        record = {
            "index": self._call_index,
            "timestamp": ts.isoformat(),
            "thread_id": thread_id,
            "node": node,
            "provider": provider,
            "model": model,
            "duration_s": duration_s,
            "usage": _resolve_usage(context, response),
            "input": [_msg_to_dict(m) for m in context],
            "output": _msg_to_dict(response),
        }
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

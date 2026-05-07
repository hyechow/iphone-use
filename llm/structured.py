"""Structured-output helpers for OpenAI-compatible chat providers."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import TypeVar

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import BadRequestError
from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)
ReturnT = TypeVar("ReturnT")
_LLM_CALL_COUNT = 0
MAX_LLM_TRANSIENT_RETRIES = 2


def get_llm_call_count() -> int:
    """Return the number of LLM API calls made through invoke_structured."""
    return _LLM_CALL_COUNT


def invoke_structured(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    schema: type[ModelT],
) -> ModelT:
    """Invoke a chat model and parse a Pydantic object.

    Uses DashScope json_object constrained decoding with thinking disabled.
    Falls back to plain JSON text parsing if the constrained mode fails.
    """
    msgs = _with_json_instruction(messages, schema)

    # Primary: json_object mode (constrained decoding) + disable thinking
    bound = llm.bind(
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )
    try:
        response = _invoke_counted_with_retry(
            lambda: bound.invoke(msgs),
            label="json_object",
        )
        content = _message_text(response.content)
        return schema.model_validate_json(_extract_json_object(content))
    except (BadRequestError, ValidationError, ValueError) as exc:
        print(f"json_object 模式失败（{type(exc).__name__}），改用纯文本 JSON 解析...")

    # Fallback: plain text, let model output JSON freely
    response = _invoke_counted_with_retry(
        lambda: llm.invoke(msgs),
        label="json text fallback",
    )
    content = _message_text(response.content)
    return schema.model_validate_json(_extract_json_object(content))


def _invoke_counted_with_retry(fn: Callable[[], ReturnT], label: str) -> ReturnT:
    """Invoke an LLM call, counting attempts and retrying transient bad payloads."""

    global _LLM_CALL_COUNT
    for attempt in range(MAX_LLM_TRANSIENT_RETRIES + 1):
        _LLM_CALL_COUNT += 1
        try:
            return fn()
        except TypeError as exc:
            if not _is_transient_response_error(exc) or attempt >= MAX_LLM_TRANSIENT_RETRIES:
                raise
            wait_s = 0.5 * (attempt + 1)
            print(f"LLM {label} 响应格式异常，{wait_s:.1f}s 后重试...")
            time.sleep(wait_s)
    raise RuntimeError("unreachable")


def _is_transient_response_error(exc: TypeError) -> bool:
    text = str(exc)
    return (
        "null value for 'choices'" in text
        or "Received response with null value for 'choices'" in text
    )


def _with_json_instruction(
    messages: list[BaseMessage],
    schema: type[BaseModel],
) -> list[BaseMessage]:
    """Merge JSON schema instruction into the system message (or prepend one)."""
    instruction = (
        "你必须只返回一个 JSON 对象，不要使用 Markdown，不要输出额外说明。\n"
        "JSON 必须符合以下 schema：\n"
        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
    )
    if messages and isinstance(messages[0], SystemMessage):
        merged = SystemMessage(content=f"{messages[0].content}\n\n{instruction}")
        return [merged, *messages[1:]]
    return [SystemMessage(content=instruction), *messages]


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    start = stripped.find("{")
    if start < 0:
        raise ValueError(f"LLM response did not contain a JSON object: {text}")
    _, end = decoder.raw_decode(stripped[start:])
    return stripped[start : start + end]

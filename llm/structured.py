"""Structured-output helpers for OpenAI-compatible chat providers."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import TypeVar

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

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

    LangChain's native structured-output parser is preferable when the provider
    implements the same response shape as OpenAI. Some compatible providers can
    return nonstandard parse payloads, so fall back to normal JSON text parsing.
    """
    try:
        return _invoke_counted_with_retry(
            lambda: llm.with_structured_output(schema).invoke(messages),
            label="structured output",
        )
    except TypeError as exc:
        if "'NoneType' object is not iterable" not in str(exc):
            raise
        print("结构化输出解析失败，改用 JSON 文本解析重试...")

    response = _invoke_counted_with_retry(
        lambda: llm.invoke(_with_json_instruction(messages, schema)),
        label="json fallback",
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
    instruction = SystemMessage(
        content=(
            "你必须只返回一个 JSON 对象，不要使用 Markdown，不要输出额外说明。\n"
            "JSON 必须符合以下 schema：\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
        )
    )
    if messages and isinstance(messages[0], SystemMessage):
        return [messages[0], instruction, *messages[1:]]
    return [instruction, *messages]


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

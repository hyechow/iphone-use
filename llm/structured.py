"""Structured-output helpers for OpenAI-compatible chat providers."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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
    primary_error: Exception | None = None
    try:
        response = _invoke_counted_with_retry(
            lambda: bound.invoke(msgs),
            label="json_object",
        )
        content = _message_text(response.content)
        return _parse_structured_response(content, schema)
    except (BadRequestError, ValidationError, ValueError) as exc:
        primary_error = exc
        print(f"json_object 模式失败（{type(exc).__name__}）: {exc}，改用纯文本 JSON 解析...")

    # Fallback: plain text, let model output JSON freely (retry once on parse failure)
    fallback_msgs = _with_repair_instruction(msgs, schema, primary_error)
    for fallback_attempt in range(2):
        response = _invoke_counted_with_retry(
            lambda: llm.invoke(fallback_msgs),
            label="json text fallback",
        )
        content = _message_text(response.content)
        try:
            return _parse_structured_response(content, schema)
        except (ValidationError, ValueError) as exc:
            if fallback_attempt == 0:
                print(f"  fallback 解析失败，重试一次...")
                fallback_msgs = _with_repair_instruction(fallback_msgs, schema, exc, content)
                continue
            raise ValueError(
                f"结构化输出解析失败（primary + fallback 均失败）: {exc}\n"
                f"模型原始输出: {content[:500]}"
            ) from exc


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
    schema_json = schema.model_json_schema()
    properties = schema_json.get("properties", {})
    required = schema_json.get("required", [])
    optional = [key for key in properties if key not in required]
    required_text = ", ".join(required) or "无"
    optional_text = ", ".join(optional) or "无"
    instruction = (
        "你必须只返回一个 JSON 对象，不要使用 Markdown，不要输出额外说明。\n"
        "返回的是业务结果实例，不是 JSON Schema。禁止返回 type/properties/description/required 这类 schema 字段作为顶层对象。\n"
        f"顶层必填字段：{required_text}\n"
        f"顶层可选字段：{optional_text}\n"
        "JSON 必须符合以下 schema（仅作为格式约束，不要照抄它）：\n"
        f"{json.dumps(schema_json, ensure_ascii=False)}"
    )
    if messages and isinstance(messages[0], SystemMessage):
        merged = SystemMessage(content=f"{messages[0].content}\n\n{instruction}")
        return [merged, *messages[1:]]
    return [SystemMessage(content=instruction), *messages]


def _with_repair_instruction(
    messages: list[BaseMessage],
    schema: type[BaseModel],
    error: Exception | None,
    raw_content: str | None = None,
) -> list[BaseMessage]:
    """Append a correction prompt after malformed structured output."""
    schema_json = schema.model_json_schema()
    required = schema_json.get("required", [])
    properties = list(schema_json.get("properties", {}).keys())
    detail = f"\n上一次错误：{error}" if error else ""
    raw = f"\n上一次原始输出片段：{raw_content[:300]}" if raw_content else ""
    instruction = (
        "请重新输出一个业务结果 JSON 对象。"
        f"顶层字段只能来自：{', '.join(properties)}。"
        f"必须包含：{', '.join(required) or '无'}。"
        "不要返回 JSON Schema，不要把 description/properties/type/required 作为顶层字段。"
        f"{detail}{raw}"
    )
    return [*messages, HumanMessage(content=instruction)]


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


def _parse_structured_response(text: str, schema: type[ModelT]) -> ModelT:
    json_text = _extract_json_object(text)
    data = json.loads(json_text)
    if _looks_like_schema_echo(data, schema):
        raise ValueError("模型返回了 JSON Schema，而不是业务结果对象")
    return schema.model_validate(data)


def _looks_like_schema_echo(data: object, schema: type[BaseModel]) -> bool:
    if not isinstance(data, dict):
        return False
    keys = set(data)
    if {"type", "properties"}.issubset(keys):
        schema_fields = set(schema.model_json_schema().get("properties", {}))
        echoed_fields = set(data.get("properties") or {})
        return not schema_fields.isdisjoint(echoed_fields)
    return False


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

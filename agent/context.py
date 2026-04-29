"""Context builder: assembles the full message list passed to the LLM.

Centralizes all context assembly logic so future memory modules and
other data sources can be plugged in here without touching agent.py.
"""
from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


class ContextBuilder:
    """Assembles the full context list for each LLM call.

    Usage::

        builder = ContextBuilder(system_message)
        context = builder.build(state["messages"])
        response = llm.invoke(context)
    """

    def __init__(
        self,
        system: SystemMessage,
        screenshot_provider: Callable[[str], str] | None = None,
    ):
        self._system = system
        self._screenshot_provider = screenshot_provider

    def build(self, state_messages: list[BaseMessage], thread_id: str | None = None) -> list[BaseMessage]:
        """Return the full context list: [system, ...processed messages].

        Currently applies screenshot image injection. Future extensions
        (long-term memory, retrieved documents, etc.) should be added here.
        """
        context, _ = self.build_with_metadata(state_messages, thread_id=thread_id)
        return context

    def build_with_metadata(
        self,
        state_messages: list[BaseMessage],
        thread_id: str | None = None,
    ) -> tuple[list[BaseMessage], str | None]:
        """Return the full context and any fresh screenshot attached to it."""
        messages = self._filter_tap_screen(state_messages)
        messages = self._inject_screenshots(messages)
        screenshot_b64 = self._take_context_screenshot(thread_id)
        if screenshot_b64:
            messages = self._attach_current_screenshot(messages, screenshot_b64)
        return [self._system] + messages, screenshot_b64

    # ── Internal transforms ────────────────────────────────────────────────

    def _take_context_screenshot(self, thread_id: str | None) -> str | None:
        if not thread_id or self._screenshot_provider is None:
            return None
        return self._screenshot_provider(thread_id)

    @staticmethod
    def _attach_current_screenshot(messages: list[BaseMessage], screenshot_b64: str) -> list[BaseMessage]:
        """Append the current screenshot as a standalone HumanMessage at the end of context."""
        return [*messages, HumanMessage(content=[
            {"type": "text", "text": "[当前屏幕截图]"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            },
        ])]

    @staticmethod
    def _filter_tap_screen(messages: list[BaseMessage]) -> list[BaseMessage]:
        """Remove tap_screen tool calls and their ToolMessage results from context.

        The LLM doesn't need to see "Tapped at (x, y)" — the next screenshot
        already shows the result. Keeps any text content in the AIMessage.
        """
        tap_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls:
                    if tc.get("name") == "tap_screen":
                        tap_ids.add(tc.get("id", ""))

        result = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.tool_call_id in tap_ids:
                continue
            if isinstance(msg, AIMessage) and msg.tool_calls:
                remaining = [tc for tc in msg.tool_calls if tc.get("name") != "tap_screen"]
                if len(remaining) != len(msg.tool_calls):
                    if not remaining and not msg.content:
                        continue
                    msg = msg.model_copy(update={"tool_calls": remaining})
            result.append(msg)
        return result

    @staticmethod
    def _inject_screenshots(messages: list[BaseMessage]) -> list[BaseMessage]:
        """Convert take_screenshot ToolMessage base64 → image_url HumanMessage.

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

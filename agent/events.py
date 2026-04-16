from dataclasses import dataclass


@dataclass
class AgentEvent:
    type: str  # "screenshot" | "thinking" | "action" | "done" | "error"
    data: str  # text, or base64-encoded PNG for "screenshot"

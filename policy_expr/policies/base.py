"""Action policy interface."""

import io
from typing import Protocol

from PIL import Image

from policy_expr.schemas import ActionDecision, Observation


class ActionPolicy(Protocol):
    """Stateless policy: maps one screenshot + one instruction to one action."""

    name: str

    def decide(self, observation: Observation, instruction: str) -> ActionDecision:
        """Return the best action for the current observation and instruction."""


class BaseActionPolicy:
    """Shared helpers for action policy implementations."""

    name = "base"

    def decide(self, observation: Observation, instruction: str) -> ActionDecision:
        raise NotImplementedError


def resize_to_logical_png(png_bytes: bytes) -> bytes:
    """Downsample Retina screenshots to logical pixels before sending to a vision model."""

    img = Image.open(io.BytesIO(png_bytes))
    logical_w, logical_h = img.width // 2, img.height // 2
    small = img.resize((logical_w, logical_h), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    return buf.getvalue()

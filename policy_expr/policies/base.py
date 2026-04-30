"""Policy interface."""

import io
import json
from pathlib import Path
from typing import Protocol

from PIL import Image

from policy_expr.schemas import Observation, PolicyContext, PolicyDecision, PolicyTurn


class Policy(Protocol):
    """A strategy that maps one observation and one goal to one action."""

    name: str

    def decide(self, observation: Observation, prompt: str) -> PolicyDecision:
        """Return the next action for the current observation."""

    def decide_with_context(
        self,
        observation: Observation,
        context: PolicyContext,
    ) -> PolicyDecision:
        """Return the next action using persisted multi-turn context."""

    def load_context(self, path: Path, prompt: str) -> PolicyContext:
        """Load or initialize persisted multi-turn context."""

    def save_context(self, path: Path, context: PolicyContext) -> None:
        """Persist multi-turn context."""

    def append_turn(
        self,
        context: PolicyContext,
        observation: Observation,
        decision: PolicyDecision,
        executed: bool,
    ) -> None:
        """Append one completed observe-decide-act turn to context."""


class BasePolicy:
    """Reusable context persistence behavior for policy implementations."""

    name = "base"

    def decide(self, observation: Observation, prompt: str) -> PolicyDecision:
        raise NotImplementedError

    def decide_with_context(
        self,
        observation: Observation,
        context: PolicyContext,
    ) -> PolicyDecision:
        return self.decide(observation, context.goal)

    def load_context(self, path: Path, prompt: str) -> PolicyContext:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return PolicyContext.model_validate(data)
        return PolicyContext(goal=prompt, policy_name=self.name)

    def save_context(self, path: Path, context: PolicyContext) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            context.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def append_turn(
        self,
        context: PolicyContext,
        observation: Observation,
        decision: PolicyDecision,
        executed: bool,
    ) -> None:
        context.turns.append(
            PolicyTurn(
                index=len(context.turns) + 1,
                observation_source=observation.source,
                screen_type=decision.screen_type,
                app_name=decision.app_name,
                summary=decision.summary,
                reasoning=decision.reasoning,
                action=decision.action,
                executed=executed,
            )
        )


def resize_to_logical_png(png_bytes: bytes) -> bytes:
    """Downsample Retina screenshots to logical pixels before sending to a vision model."""

    img = Image.open(io.BytesIO(png_bytes))
    logical_w, logical_h = img.width // 2, img.height // 2
    small = img.resize((logical_w, logical_h), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    return buf.getvalue()

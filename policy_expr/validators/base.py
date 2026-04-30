"""Validator interface for policy experiments."""

from typing import Protocol
from typing import Optional

from pydantic import BaseModel, Field

from policy_expr.schemas import Observation, PolicyDecision


class ValidationResult(BaseModel):
    """Result of checking whether a policy turn achieved the expected effect."""

    passed: bool = Field(description="验证是否通过")
    summary: str = Field(description="验证结果说明")
    evidence: Optional[str] = Field(default=None, description="用于判断的关键视觉证据")


class Validator(Protocol):
    """Checks the result of an observe-decide-act turn."""

    name: str

    def validate(
        self,
        before: Observation,
        decision: PolicyDecision,
        after: Observation,
    ) -> ValidationResult:
        """Return whether the turn outcome looks correct."""

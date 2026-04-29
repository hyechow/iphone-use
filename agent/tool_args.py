"""Normalize model-produced tool arguments before execution/logging."""
from typing import Any


def _is_scalar_like(value: Any) -> bool:
    if isinstance(value, list):
        return len(value) == 1
    if isinstance(value, dict):
        return "value" in value
    return True


def coerce_number(value: Any, default: float = 0) -> float:
    """Coerce scalar-like model output into a number."""
    if isinstance(value, list):
        return coerce_number(value[0], default=default) if len(value) == 1 else default
    if isinstance(value, dict):
        if "value" in value:
            return coerce_number(value["value"], default=default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_coordinate(value: Any) -> Any:
    if _is_scalar_like(value):
        return coerce_number(value)
    return value



def _normalize_tap_args(args: dict[str, Any]) -> dict[str, Any]:
    x = args.get("x")
    y = args.get("y")

    # Model packed both coords into x as [x, y] with no y field.
    if isinstance(x, list) and len(x) == 2 and y is None:
        return {**args, "x": coerce_number(x[0]), "y": coerce_number(x[1])}

    # Model passed x as [a, b] and y as [c]. Take x[0] as x and y[0] as y.
    if (
        isinstance(x, list)
        and len(x) == 2
        and isinstance(y, list)
        and len(y) == 1
    ):
        return {**args, "x": coerce_number(x[0]), "y": coerce_number(y[0])}

    return {**args, "x": _normalize_coordinate(x), "y": _normalize_coordinate(y)}


def normalize_tool_args(name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Return normalized args for known tools."""
    normalized = dict(args or {})
    if name == "tap_screen":
        normalized = _normalize_tap_args(normalized)
    return normalized

"""Supervisor policy implementations."""

__all__ = ["MilestoneSupervisorPolicy", "SimpleSupervisorPolicy"]


def __getattr__(name: str):
    if name == "MilestoneSupervisorPolicy":
        from policy_expr.supervisor.milestone import MilestoneSupervisorPolicy
        return MilestoneSupervisorPolicy
    if name == "SimpleSupervisorPolicy":
        from policy_expr.supervisor.simple import SimpleSupervisorPolicy
        return SimpleSupervisorPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

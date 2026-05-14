"""BFS exploration trace: records page visits, navigation structure, and back-navigation attempts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class BackAttempt:
    strategy: str           # "fixed_1", "YOLO", "LLM", "LLM+YOLO"
    result: str             # "匹配" | "不匹配" | "未变化" | "父页面"
    score: float = 0.0
    coords: list[int] = field(default_factory=list)  # [x, y] logical coords
    success: bool = False


@dataclass
class ProbeError:
    message: str
    failed_tap: int = -1      # tap index that caused failure; -1 = end-of-probe cleanup
    failed_element: str = ""
    back_attempts: list[BackAttempt] = field(default_factory=list)


@dataclass
class BfsEntry:
    page: str
    parent: str | None
    via_tap: str | None
    depth: int
    error: ProbeError | None = None


class BfsTracer:
    """Accumulates BFS exploration entries and persists them incrementally to disk."""

    def __init__(self) -> None:
        self._entries: list[BfsEntry] = []
        self._page_index: dict[str, int] = {}

    def record_page(self, page: str, parent: str | None, via_tap: str | None, depth: int) -> None:
        """Record that BFS has started exploring a page."""
        entry = BfsEntry(page=page, parent=parent, via_tap=via_tap, depth=depth)
        self._page_index[page] = len(self._entries)
        self._entries.append(entry)

    def record_error(self, page: str, exc: BaseException) -> None:
        """Attach a structured or plain-string error to the last recorded entry for page."""
        from policy_expr.recon.utils import ProbeAbortedError
        idx = self._page_index.get(page)
        if idx is None:
            return
        if isinstance(exc, ProbeAbortedError):
            self._entries[idx].error = ProbeError(
                message=str(exc),
                failed_tap=exc.failed_tap,
                failed_element=exc.failed_element,
                back_attempts=[BackAttempt(**a) for a in exc.back_attempts],
            )
        else:
            self._entries[idx].error = ProbeError(message=str(exc))

    @property
    def entries(self) -> list[BfsEntry]:
        return list(self._entries)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._to_list(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _to_list(self) -> list[dict]:
        result = []
        for e in self._entries:
            d: dict = {"page": e.page, "parent": e.parent, "via_tap": e.via_tap, "depth": e.depth}
            if e.error:
                err = e.error
                d["error"] = {
                    "message": err.message,
                    "failed_tap": err.failed_tap,
                    "failed_element": err.failed_element,
                    "back_attempts": [asdict(a) for a in err.back_attempts],
                }
            result.append(d)
        return result

    @classmethod
    def load(cls, path: Path) -> "BfsTracer":
        tracer = cls()
        if not path.exists():
            return tracer
        data = json.loads(path.read_text(encoding="utf-8"))
        for d in data:
            page = d.get("page", "")
            tracer._page_index[page] = len(tracer._entries)
            error: ProbeError | None = None
            raw_err = d.get("error")
            if raw_err:
                if isinstance(raw_err, str):
                    error = ProbeError(message=raw_err)
                else:
                    error = ProbeError(
                        message=raw_err.get("message", ""),
                        failed_tap=raw_err.get("failed_tap", -1),
                        failed_element=raw_err.get("failed_element", ""),
                        back_attempts=[BackAttempt(**a) for a in raw_err.get("back_attempts", [])],
                    )
            tracer._entries.append(BfsEntry(
                page=page,
                parent=d.get("parent"),
                via_tap=d.get("via_tap"),
                depth=d.get("depth", 0),
                error=error,
            ))
        return tracer

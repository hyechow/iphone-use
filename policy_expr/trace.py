"""Page exploration trace: records page visits, navigation structure, and back-navigation attempts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class BackAttempt:
    strategy: str           # "fixed_1", "YOLO", "LLM", "LLM+YOLO", "retry"
    result: str             # "initial" | "L0" | "L1" | "未知页" | "未变化" | "forward"
    score: float = 0.0
    coords: list[int] = field(default_factory=list)  # [x, y] logical coords
    success: bool = False
    screenshot: str = ""    # path to screenshot after this attempt, or ""


@dataclass
class ProbeError:
    message: str
    failed_tap: int = -1      # tap index that caused failure; -1 = end-of-probe cleanup
    failed_element: str = ""
    back_attempts: list[BackAttempt] = field(default_factory=list)


@dataclass
class Entry:
    page: str
    parent: str | None
    via_tap: str | None
    depth: int
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    error: ProbeError | None = None


@dataclass
class Transition:
    src: str        # 来源页面名
    tap: str        # 触发跳转的元素标签
    dst: str        # 目标页面名
    # "entered"=进入探测 | "depth_limit"=仅记录 | "skipped_visited"=已访问跳过 | "skipped_known"=已学跳过
    status: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))


class Tracer:
    """Accumulates page exploration entries and persists them incrementally to disk."""

    def __init__(self) -> None:
        self._entries: list[Entry] = []
        self._page_index: dict[str, int] = {}
        self._transitions: list[Transition] = []

    def record_page(self, page: str, parent: str | None, via_tap: str | None, depth: int) -> None:
        """Record that exploration has started exploring a page."""
        entry = Entry(page=page, parent=parent, via_tap=via_tap, depth=depth)
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

    def record_transition(self, src: str, tap: str, dst: str, status: str) -> None:
        self._transitions.append(Transition(src=src, tap=tap, dst=dst, status=status))

    @property
    def entries(self) -> list[Entry]:
        return list(self._entries)

    @property
    def transitions(self) -> list[Transition]:
        return list(self._transitions)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pages": self._to_list(),
            "transitions": [asdict(t) for t in self._transitions],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _to_list(self) -> list[dict]:
        result = []
        for e in self._entries:
            d: dict = {"page": e.page, "parent": e.parent, "via_tap": e.via_tap, "depth": e.depth, "timestamp": e.timestamp}
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
    def load(cls, path: Path) -> "Tracer":
        tracer = cls()
        if not path.exists():
            return tracer
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Support both old format (list) and new format (dict with pages/transitions)
        if isinstance(raw, list):
            entries_data, transitions_data = raw, []
        else:
            entries_data = raw.get("pages", [])
            transitions_data = raw.get("transitions", [])
        for t in transitions_data:
            tracer._transitions.append(Transition(**t))
        data = entries_data
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
            tracer._entries.append(Entry(
                page=page,
                parent=d.get("parent"),
                via_tap=d.get("via_tap"),
                depth=d.get("depth", 0),
                timestamp=d.get("timestamp", ""),
                error=error,
            ))
        return tracer

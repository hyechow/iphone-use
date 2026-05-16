"""Stateful page library for exploration dedup: semantic-primary cascade.

Cascade logic (OR):
  1. Visual embedding (GUIClip, fast, no LLM).
     - If max_visual >= visual_shortcut → duplicate (save LLM call).
  2. Text embedding (bge-small-zh via LLM fingerprint).
     - If max_text >= text_threshold → duplicate.
     - Otherwise → new page, add to library.

Semantic is the primary verdict signal; visual is used as an efficiency
shortcut for the clear-match case.  Thresholds are calibrated from the
微信 dataset experiment (tv=0.85→0.95 for shortcut, tt=0.92).
"""

from __future__ import annotations

from dataclasses import dataclass

from policy_expr.recon.cascade_matcher import CascadeMatcher, PageEmbedding


@dataclass
class DedupResult:
    is_duplicate: bool
    reason: str
    best_visual_sim: float = 0.0
    best_text_sim: float | None = None


class PageDedup:
    """Semantic-primary page dedup for BFS/DFS exploration.

    Usage::

        matcher = CascadeMatcher()
        dedup = PageDedup(matcher)
        result = dedup.check_and_add(screenshot_png)
        if result.is_duplicate:
            skip_this_page()
    """

    def __init__(
        self,
        matcher: CascadeMatcher,
        text_threshold: float = 0.92,
        visual_shortcut: float = 0.95,
    ):
        self._matcher = matcher
        self._tt = text_threshold
        self._vs = visual_shortcut
        self._library: list[tuple[PageEmbedding, bytes]] = []

    def __len__(self) -> int:
        return len(self._library)

    def check(self, png: bytes) -> DedupResult:
        """Check if png is a duplicate of any page in the library.

        Does NOT modify the library.
        """
        if not self._library:
            return DedupResult(False, "empty_library")

        # Phase 1: visual (fast, no LLM)
        candidate = self._matcher.embed_visual(png)
        vis_sims = [self._matcher.visual_sim(candidate, e) for e, _ in self._library]
        max_vis = max(vis_sims)

        if max_vis >= self._vs:
            return DedupResult(True, f"visual_shortcut({max_vis:.3f})", max_vis)

        # Phase 2: semantic text (LLM call for candidate + any un-embedded library pages)
        self._matcher.fill_text(candidate, png)
        for emb, epng in self._library:
            self._matcher.fill_text(emb, epng)

        txt_sims = [self._matcher.text_sim(candidate, e) for e, _ in self._library]
        max_txt = max(txt_sims)

        if max_txt >= self._tt:
            return DedupResult(True, f"text_match({max_txt:.3f})", max_vis, max_txt)

        return DedupResult(
            False, f"no_match(vis={max_vis:.3f},txt={max_txt:.3f})", max_vis, max_txt
        )

    def add(self, png: bytes, emb: PageEmbedding | None = None) -> PageEmbedding:
        """Add page to library and return its embedding."""
        if emb is None:
            emb = self._matcher.embed_visual(png)
        self._library.append((emb, png))
        return emb

    def check_and_add(self, png: bytes) -> DedupResult:
        """Check for duplicate; add to library only if not a duplicate."""
        result = self.check(png)
        if not result.is_duplicate:
            self.add(png)
        return result

"""Page identity: determine whether a page is new or already known.

Cascade logic (OR):
  1. Visual embedding (GUIClip, fast, no LLM).
     - If max_visual >= visual_shortcut → known page (save LLM call).
  2. Text embedding (bge-small-zh via LLM fingerprint).
     - If max_text >= text_threshold → known page.
     - Otherwise → new page, add to library.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from policy_expr.recon.cascade_matcher import CascadeMatcher, PageEmbedding


@dataclass
class IdentityResult:
    is_duplicate: bool
    reason: str
    best_visual_sim: float = 0.0
    best_text_sim: float | None = None
    matched_index: int | None = None
    matched_name: str | None = None
    library_size: int = 0
    phase: str = ""  # "visual_shortcut" / "text_match" / "new_page"


class PageIdentity:
    """Semantic-primary page identity for BFS/DFS exploration."""

    def __init__(
        self,
        matcher: CascadeMatcher,
        text_threshold: float = 0.92,
        visual_shortcut: float = 0.95,
    ):
        self._matcher = matcher
        self._tt = text_threshold
        self._vs = visual_shortcut
        self._library: list[tuple[PageEmbedding, bytes, str]] = []

    def __len__(self) -> int:
        return len(self._library)

    def check(self, png: bytes, precomputed: PageEmbedding | None = None) -> IdentityResult:
        """Check if png is a known page in the library."""
        n = len(self._library)
        if n == 0:
            return IdentityResult(False, "empty_library", library_size=0, phase="new_page")

        # Phase 1: visual (fast, no LLM)
        candidate = precomputed or self._matcher.embed_visual(png)
        vis_sims = [self._matcher.visual_sim(candidate, e) for e, _, _ in self._library]
        best_idx = max(range(n), key=lambda i: vis_sims[i])
        max_vis = vis_sims[best_idx]

        if max_vis >= self._vs:
            return IdentityResult(
                True, f"visual_shortcut({max_vis:.3f})",
                max_vis, matched_index=best_idx,
                matched_name=self._library[best_idx][2],
                library_size=n, phase="visual_shortcut",
            )

        # Phase 2: semantic text (LLM call for candidate + un-embedded library pages)
        self._matcher.fill_text(candidate, png)
        for emb, epng, _ in self._library:
            self._matcher.fill_text(emb, epng)

        txt_sims = [self._matcher.text_sim(candidate, e) for e, _, _ in self._library]
        txt_best_idx = max(range(n), key=lambda i: txt_sims[i])
        max_txt = txt_sims[txt_best_idx]

        if max_txt >= self._tt:
            return IdentityResult(
                True, f"text_match({max_txt:.3f})",
                max_vis, max_txt, matched_index=txt_best_idx,
                matched_name=self._library[txt_best_idx][2],
                library_size=n, phase="text_match",
            )

        return IdentityResult(
            False, f"no_match(vis={max_vis:.3f},txt={max_txt:.3f})",
            max_vis, max_txt, library_size=n, phase="new_page",
        )

    def add(self, png: bytes, name: str = "", emb: PageEmbedding | None = None) -> PageEmbedding:
        """Add page to library and return its embedding."""
        if emb is None:
            emb = self._matcher.embed_visual(png)
        self._library.append((emb, png, name))
        return emb

    def check_and_add(self, png: bytes, name: str = "") -> IdentityResult:
        """Check for identity; add to library only if not a known page."""
        result = self.check(png)
        if not result.is_duplicate:
            self.add(png, name)
        return result

    def to_json(self) -> list[dict]:
        """Export library entries as JSON-serializable list."""
        return [
            {"index": i, "name": name}
            for i, (_, _, name) in enumerate(self._library)
        ]

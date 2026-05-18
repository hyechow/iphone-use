"""Page comparison: unified API for screenshot similarity and navigation detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from policy_expr.recon.utils import ScreenMatchDecision, png_similarity

if TYPE_CHECKING:
    from policy_expr.recon.cascade_matcher import CascadeMatcher


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SimilarityBackend(Protocol):
    """Compute a 0-1 similarity score between two PNG screenshots."""

    def similarity(self, png_a: bytes, png_b: bytes) -> float: ...


class EdgeIoUBackend:
    """Edge-IoU similarity (Canny edge overlap). Zero dependencies beyond skimage."""

    def similarity(self, png_a: bytes, png_b: bytes) -> float:
        return png_similarity(png_a, png_b)


# ---------------------------------------------------------------------------
# PageComparator
# ---------------------------------------------------------------------------

class PageComparator:
    """High-level page comparison backed by a pluggable SimilarityBackend.

    Navigation detection uses a visual-primary cascade:
      1. Fast EdgeIoU score.
         - >= no_change_threshold  → same page (no navigation)
         - <  same_page_threshold  → navigated
      2. Gray zone: use CascadeMatcher (GUIClip + semantic) to disambiguate.
         - GUIClip sim >= cascade_vis  → same page type; confirm with text
         - text sim >= cascade_txt     → same page (content refresh, not nav)
         - otherwise                   → navigated
    """

    def __init__(
        self,
        backend: SimilarityBackend | None = None,
        same_page_threshold: float = 0.20,
        no_change_threshold: float = 0.80,
        cascade: "CascadeMatcher | None" = None,
        cascade_vis: float = 0.85,
        cascade_txt: float = 0.92,
    ):
        self._backend = backend or EdgeIoUBackend()
        self._same_page_threshold = same_page_threshold
        self._no_change_threshold = no_change_threshold
        self._cascade = cascade
        self._cascade_vis = cascade_vis
        self._cascade_txt = cascade_txt

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__

    # --- low-level ----------------------------------------------------------

    def raw_similarity(self, png_a: bytes, png_b: bytes) -> float:
        return self._backend.similarity(png_a, png_b)

    # --- Scenario A: same page ----------------------------------------------

    def is_same_page(
        self,
        reference_png: bytes,
        candidate_png: bytes | None,
    ) -> ScreenMatchDecision:
        if not candidate_png:
            return ScreenMatchDecision(False, 0.0, self.backend_name, "missing screenshot")
        sim = self.raw_similarity(reference_png, candidate_png)
        matched = sim >= self._same_page_threshold
        return ScreenMatchDecision(
            matched, sim, self.backend_name,
            f"similarity {sim:.3f} {'above' if matched else 'below'} "
            f"same-page threshold {self._same_page_threshold}",
        )

    # --- Scenario B: no change ----------------------------------------------

    def is_unchanged(self, before_png: bytes, after_png: bytes) -> bool:
        return self.raw_similarity(before_png, after_png) >= self._no_change_threshold

    def no_change_score(self, before_png: bytes, after_png: bytes) -> tuple[bool, float]:
        sim = self.raw_similarity(before_png, after_png)
        return sim >= self._no_change_threshold, sim

    # --- Scenario C: direction / progress -----------------------------------

    def is_closer_to(
        self,
        candidate_png: bytes,
        reference_png: bytes,
        baseline_png: bytes,
    ) -> bool:
        return (self.raw_similarity(reference_png, candidate_png)
                > self.raw_similarity(reference_png, baseline_png))

    def similarity_to(self, reference_png: bytes, candidate_png: bytes) -> float:
        return self.raw_similarity(reference_png, candidate_png)

    # --- Scenario D: navigation detection -----------------------------------

    def detect_navigation(
        self,
        initial_png: bytes,
        after_png: bytes | None,
    ) -> tuple[bool, str]:
        """Did tapping an element navigate to a different page?

        Returns (navigated, reason).
        """
        if not after_png:
            return False, "no screenshot"

        sim = self.raw_similarity(initial_png, after_png)

        if sim >= self._no_change_threshold:
            return False, f"no_change (sim={sim:.3f})"

        if sim < self._same_page_threshold:
            return True, f"navigated (sim={sim:.3f})"

        # Gray zone: use cascade matcher to disambiguate
        if self._cascade is not None:
            return _cascade_nav_check(
                initial_png, after_png, sim, self._cascade,
                self._cascade_vis, self._cascade_txt,
            )

        return True, f"navigated (sim={sim:.3f})"


# ---------------------------------------------------------------------------
# Cascade nav helper (visual-primary)
# ---------------------------------------------------------------------------

def _cascade_nav_check(
    initial_png: bytes,
    after_png: bytes,
    edge_sim: float,
    cascade: "CascadeMatcher",
    vis_threshold: float,
    txt_threshold: float,
) -> tuple[bool, str]:
    """Visual-primary cascade for navigation disambiguation."""
    emb_a = cascade.embed_visual(initial_png)
    emb_b = cascade.embed_visual(after_png)
    vis = cascade.visual_sim(emb_a, emb_b)

    if vis < vis_threshold:
        return True, f"navigated (edge={edge_sim:.3f}, guiclip={vis:.3f})"

    # GUIClip says similar → confirm with semantic
    cascade.fill_text(emb_a, initial_png)
    cascade.fill_text(emb_b, after_png)
    txt = cascade.text_sim(emb_a, emb_b)

    if txt >= txt_threshold:
        return False, f"same_page (guiclip={vis:.3f}, text={txt:.3f})"

    return True, f"navigated (guiclip={vis:.3f}, text={txt:.3f})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_comparator(method: str = "edge_iou") -> PageComparator:
    """Construct a PageComparator by method name.

    Args:
        method: "edge_iou" (default), "guiclip", "cascade".
    """
    if method == "guiclip":
        from policy_expr.recon.guiclip_backend import GUIClipBackend
        return PageComparator(
            backend=GUIClipBackend(),
            same_page_threshold=0.90,
            no_change_threshold=0.98,
        )
    if method == "cascade":
        from policy_expr.recon.cascade_matcher import get_matcher
        cascade = get_matcher()
        return PageComparator(cascade=cascade)
    return PageComparator()

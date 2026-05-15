"""Page comparison: unified API for screenshot similarity and navigation detection."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from policy_expr.recon.utils import ScreenMatchDecision, png_similarity


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
    """High-level page comparison backed by a pluggable SimilarityBackend."""

    def __init__(
        self,
        backend: SimilarityBackend | None = None,
        same_page_threshold: float = 0.20,
        no_change_threshold: float = 0.80,
        use_fingerprint: bool = True,
    ):
        self._backend = backend or EdgeIoUBackend()
        self._same_page_threshold = same_page_threshold
        self._no_change_threshold = no_change_threshold
        self._use_fingerprint = use_fingerprint

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

        if sim >= self._same_page_threshold:
            if self._use_fingerprint and _fingerprint_match(initial_png, after_png):
                return False, f"same_page (sim={sim:.3f}, fingerprint match)"

        return True, f"navigated (sim={sim:.3f})"


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------

def _fingerprint_match(png_a: bytes, png_b: bytes) -> bool:
    from policy_expr.recon.fingerprint import compute_fingerprint
    return compute_fingerprint(png_a).key == compute_fingerprint(png_b).key


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_comparator(method: str = "edge_iou") -> PageComparator:
    """Construct a PageComparator by method name.

    Args:
        method: "edge_iou" (default), "guiclip".
    """
    if method == "guiclip":
        from policy_expr.recon.guiclip_backend import GUIClipBackend
        return PageComparator(
            backend=GUIClipBackend(),
            same_page_threshold=0.90,
            no_change_threshold=0.98,
        )
    return PageComparator()

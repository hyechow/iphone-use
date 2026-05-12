"""Reconnaissance mode: parse app UI screenshots into structured page knowledge."""

from policy_expr.recon.fingerprint import PageFingerprint, compute_fingerprint
from policy_expr.recon.page_parser import (
    InteractiveArea,
    InteractiveElement,
    PageKnowledge,
    PageParser,
    ParsedPage,
)
from policy_expr.recon.utils import print_areas, viz_result, visualize, visualize_areas

__all__ = [
    "PageParser", "ParsedPage", "InteractiveElement",
    "PageKnowledge", "InteractiveArea",
    "PageFingerprint", "compute_fingerprint",
    "visualize", "visualize_areas", "print_areas", "viz_result",
]

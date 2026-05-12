"""Breadth-first page exploration: tap each element, capture after-state."""

from __future__ import annotations

import time
from pathlib import Path

from policy_expr.recon.page_parser import ParsedPage
from policy_expr.recon.utils import ReconResult, TapResult


def probe_elements(client, page: ParsedPage, out_dir: Path) -> ReconResult:
    """Tap each semantic element, capture after-state, return structured result.

    For each element:
    1. Tap at its logical coordinate
    2. Screenshot the after-state
    3. Go back (if element might have navigated away)
    """
    from policy_expr.executor import logical_xy

    targets = [
        el for el in page.interactive_elements
        if el.label or el.element_type == "back_button"
    ]
    print(f"\n{'=' * 60}")
    print(f"点击探测: {len(targets)} 个语义元素")
    print(f"{'=' * 60}")

    has_nav = page.bottom_nav.has_nav
    if has_nav:
        print("  检测到底部导航栏，tab 元素点击后不返回")

    ident = page.identity
    result = ReconResult(
        app_name=ident.app_name,
        page_title=ident.page_title,
        page_type=ident.page_type,
        signature=ident.signature,
        description=page.description,
        elements_count=len(page.interactive_elements),
    )

    for i, el in enumerate(targets, 1):
        label = el.label or el.element_type
        lx, ly = logical_xy(el.x, el.y)
        print(f"\n  [{i}/{len(targets)}] 「{label}」 @ ({el.x:.0f},{el.y:.0f}) → ({lx:.0f},{ly:.0f})")

        tap_response = client.tap(lx, ly)
        tap_ok = "failed" not in tap_response.lower() and "interrupted" not in tap_response.lower()
        print(f"    结果: {tap_response}")
        time.sleep(2.0)

        after_bytes = client.screenshot() if hasattr(client, "screenshot") else b""
        after_path = out_dir / f"tap_{i:02d}_{el.element_type}.png"
        if after_bytes:
            after_path.write_bytes(after_bytes)
            print(f"    截图: {after_path}")

        navigated = el.element_type in ("link", "button", "menu_item")

        result.taps.append(TapResult(
            index=i,
            element_type=el.element_type,
            label=label,
            x=el.x,
            y=el.y,
            tap_ok=tap_ok,
            screenshot_path=str(after_path),
            navigated=navigated,
        ))

        # Skip "go back" for: back_button, or tab elements when bottom nav exists
        is_nav_tab = has_nav and el.element_type == "tab"
        if el.element_type not in ("tab", "back_button") and not is_nav_tab:
            print("    返回...")
            blx, bly = logical_xy(85, 147)
            client.tap(blx, bly)
            time.sleep(1.5)

    return result

"""Breadth-first page exploration: tap each element, capture after-state."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from policy_expr.perception import try_resume_mac
from policy_expr.recon.page_compare import PageComparator, make_comparator
from policy_expr.recon.page_parser import PageKnowledge
from policy_expr.recon.back_nav import (
    return_to_initial, back_shot_path, save_if_changed, _match_stack,
)
from policy_expr.recon.utils import (
    ProbeAbortedError,
    ReconResult,
    TapResult,
)

# Module-level comparator (lazy, edge IoU by default)
_comparator: PageComparator | None = None


def _get_comparator() -> PageComparator:
    global _comparator
    if _comparator is None:
        _comparator = make_comparator()
    return _comparator


def probe_elements(
    client,
    knowledge: PageKnowledge,
    out_dir: Path,
    initial_screenshot_path: Path | None = None,
    screenshot: Callable[[], bytes] | None = None,
    debug: bool = False,
    sample: int = 0,
    nav_stack: list[tuple[bytes, tuple[float, float] | None]] | None = None,
) -> ReconResult:
    """Tap each area, capture after-state, return structured result.

    Args:
        sample: If > 0, randomly sample this many elements instead of probing all.
        nav_stack: Navigation stack [(page_bytes, forward_coords), ...].
                   Last entry is the initial page. If None, built from initial screenshot.
    """
    import random
    from policy_expr.executor import logical_xy

    page = knowledge.page
    areas = knowledge.areas
    has_nav = page.bottom_nav.has_nav

    # Back-button taps always navigate to parent — skip, their behavior is known.
    areas = [a for a in areas if a.element_type != "back_button"]

    if sample > 0 and sample < len(areas):
        areas = random.sample(areas, sample)
        print(f"  [采样模式] 随机选取 {sample} 个元素")

    print(f"\n{'=' * 60}")
    print(f"点击探测: {len(areas)} 个可交互区域")
    print(f"{'=' * 60}")

    ident = page.identity
    result = ReconResult(
        app_name=ident.app_name,
        page_title=ident.page_title,
        page_type=ident.page_type,
        signature=ident.signature,
        description=page.description,
        elements_count=len(page.interactive_elements),
        initial_screenshot_path=str(initial_screenshot_path or ""),
    )

    initial_bytes = None
    if initial_screenshot_path and initial_screenshot_path.exists():
        initial_bytes = initial_screenshot_path.read_bytes()
    if screenshot is None:
        raise ValueError("probe_elements requires an SCK screenshot callable")

    # Build nav_stack from initial screenshot if not provided
    if nav_stack is None:
        nav_stack = [(initial_bytes, None)] if initial_bytes else []
    top_level = len(nav_stack) - 1

    tap_dir = out_dir / "tap"
    tap_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "recon_result.json"

    for i, area in enumerate(areas, 1):
        ax, ay = area.center_xy
        lx, ly = logical_xy(ax, ay)
        is_tab = has_nav and ay > 900
        print(f"\n  [{i}/{len(areas)}] 「{area.label}」 @ ({ax:.0f},{ay:.0f}) → ({lx:.0f},{ly:.0f})")

        tap_response = client.tap(lx, ly)
        if "paused" in tap_response.lower():
            print(f"    Mac 弹窗阻断，关闭后跳过")
            try_resume_mac()
            result.taps.append(TapResult(
                index=i, element_type="tab" if is_tab else "area",
                label=area.label, x=ax, y=ay,
                tap_ok=True, screenshot_path="", navigated=False,
            ))
            result.save(result_path)
            continue
        tap_ok = "failed" not in tap_response.lower() and "interrupted" not in tap_response.lower()
        print(f"    结果: {tap_response}")
        time.sleep(2.0)

        after_bytes = screenshot()
        after_path = tap_dir / f"tap_{i:02d}_{area.label}.png"
        if after_bytes:
            after_path.write_bytes(after_bytes)
            print(f"    截图: {after_path}")

        navigated = False
        if after_bytes and nav_stack:
            comp = _get_comparator()
            nav_result, nav_reason = comp.detect_navigation(nav_stack[top_level][0], after_bytes)
            if not nav_result:
                print(f"    {nav_reason}，跳过")
            else:
                navigated = True

        result.taps.append(TapResult(
            index=i,
            element_type="tab" if is_tab else "area",
            label=area.label,
            x=ax,
            y=ay,
            tap_ok=tap_ok,
            screenshot_path=str(after_path),
            navigated=navigated,
        ))

        if navigated and nav_stack:
            matched, back_log = return_to_initial(
                client, screenshot, nav_stack,
                before_back_bytes=after_bytes,
                out_dir=tap_dir,
                tap_index=i,
                nav_context=f"点击了「{area.label}」",
            )
            result.taps[-1].back_attempts = back_log
            result.save(result_path)
            if not matched:
                raise ProbeAbortedError(
                    f"无法返回子页面（探测第 {i} 个元素「{area.label}」后），终止探测",
                    failed_tap=i,
                    failed_element=area.label,
                    back_attempts=back_log,
                )

        result.save(result_path)

        if debug:
            input("    [DEBUG] 按回车继续下一个区域...")

    # Ensure we return to initial page after probing
    if initial_bytes:
        current_bytes = screenshot()
        comp = _get_comparator()
        matched_level = _match_stack(comp, nav_stack, current_bytes)
        if matched_level < 0 or matched_level != top_level:
            print("  [end-of-probe] 需要返回初始页面")
            matched, back_log = return_to_initial(
                client, screenshot, nav_stack,
                before_back_bytes=current_bytes,
                out_dir=tap_dir,
                tap_index=0,
            )
            if not matched:
                raise ProbeAbortedError(
                    "探测完成后无法返回初始页面，终止",
                    failed_tap=-1,
                    failed_element="",
                    back_attempts=back_log,
                )

    return result

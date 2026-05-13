"""Breadth-first page exploration: tap each element, capture after-state."""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm.structured import invoke_structured
from policy_expr.config import resolve_llm_config
from policy_expr.policies.base import resize_to_logical_png
from policy_expr.recon.page_parser import ParsedPage, PageKnowledge
from policy_expr.recon.utils import (
    ReconResult,
    ScreenMatchDecision,
    TapResult,
    SCREEN_MATCH_THRESHOLD,
    png_similarity,
)


BACK_TAP_CENTER = (70.0, 125.0)
BACK_TAP_JITTER = (3.0, 3.0)
BACK_SETTLE_SECONDS = 1.5
BACK_ACTION_NO_CHANGE_THRESHOLD = 0.995
BACK_MATCH_THRESHOLD = 0.989
BACK_TAP_POINTS = (
    BACK_TAP_CENTER,
    (45.0, 125.0),
)
BACK_SWIPE_START = (12.0, 500.0)
BACK_SWIPE_END = (620.0, 500.0)


class BackAction(BaseModel):
    can_go_back: bool = Field(description="能否找到返回到前一页的方法")
    method: str = Field(description="返回方法描述")
    back_x: float = Field(default=-1, description="返回目标归一化 x 坐标（0-1000）")
    back_y: float = Field(default=-1, description="返回目标归一化 y 坐标（0-1000）")


BACK_PROMPT = """\
你是一个手机页面导航分析器。

用户给出了两张截图：
- 第一张（BEFORE）：点击前的原始页面
- 第二张（AFTER）：点击某个元素后跳转到的页面

请分析 AFTER 页面上有什么方法可以返回到 BEFORE 页面。

常见的返回方式：
1. 左上角返回按钮（< 箭头）→ 点击它
2. 底部 tab 栏中的某个 tab → 点击对应 tab
3. 关闭按钮（×）→ 点击它

输出：
- can_go_back: 是否能找到返回方法
- method: 描述返回方法
- back_x, back_y: 返回目标坐标（0-1000，左上角原点）
"""


def _is_navigation_target(el, has_nav: bool) -> bool:
    return el.element_type == "back_button" or (has_nav and el.element_type == "tab")

def _matches_initial_layered(
    initial_page: ParsedPage,
    initial_png: bytes,
    current_png: bytes | None,
    initial_fingerprint: str | None = None,
) -> ScreenMatchDecision:
    if not current_png:
        return ScreenMatchDecision(False, 0.0, "screenshot", "missing current screenshot")

    # Level 1: pixel similarity (fast, free)
    similarity = png_similarity(initial_png, current_png)
    if similarity >= BACK_MATCH_THRESHOLD:
        return ScreenMatchDecision(
            True,
            similarity,
            "pixel",
            f"similarity above back match threshold {BACK_MATCH_THRESHOLD}",
        )

    # Level 2: fingerprint comparison (one LLM call)
    from policy_expr.recon.fingerprint import compute_fingerprint

    if initial_fingerprint is None:
        initial_fingerprint = compute_fingerprint(initial_png).key
    current_fingerprint = compute_fingerprint(current_png).key
    matched = initial_fingerprint == current_fingerprint
    reason = f"fingerprint {'match' if matched else 'mismatch'}: [{current_fingerprint}]"
    return ScreenMatchDecision(matched, similarity, "fingerprint", reason)



def _back_tap_point(attempt: int) -> tuple[float, float]:
    point = BACK_TAP_POINTS[(attempt - 1) % len(BACK_TAP_POINTS)]
    if attempt <= len(BACK_TAP_POINTS):
        return point
    return (
        point[0] + np.random.uniform(-BACK_TAP_JITTER[0], BACK_TAP_JITTER[0]),
        point[1] + np.random.uniform(-BACK_TAP_JITTER[1], BACK_TAP_JITTER[1]),
    )


def tap_back(client, attempt: int = 1) -> tuple[float, float, str]:
    """Tap near the iOS back button."""
    from policy_expr.executor import logical_xy

    ax, ay = _back_tap_point(attempt)
    lx, ly = logical_xy(ax, ay)
    result = client.tap(lx, ly)
    return lx, ly, result


def tap_llm_back(client, action: BackAction) -> tuple[float, float, str]:
    """Tap the return target selected by the vision model."""
    from policy_expr.executor import logical_xy

    lx, ly = logical_xy(action.back_x, action.back_y)
    result = client.tap(lx, ly)
    return lx, ly, result


def swipe_back(client) -> tuple[tuple[float, float], tuple[float, float], str]:
    """Use the iOS left-edge back gesture as a final fallback."""
    from policy_expr.executor import logical_xy

    start_x, start_y = logical_xy(*BACK_SWIPE_START)
    end_x, end_y = logical_xy(*BACK_SWIPE_END)
    if not hasattr(client, "swipe"):
        return (start_x, start_y), (end_x, end_y), "swipe unavailable"
    result = client.swipe(start_x, start_y, end_x, end_y, duration_ms=450)
    return (start_x, start_y), (end_x, end_y), result


BACK_MAX_RETRIES = len(BACK_TAP_POINTS)


def infer_back_action(before_png: bytes, after_png: bytes | None) -> BackAction | None:
    """Ask the vision model how to navigate from AFTER back to BEFORE."""
    if not after_png:
        return None

    cfg = resolve_llm_config("action_policy")
    llm = ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=0,
    )
    before_b64 = base64.b64encode(resize_to_logical_png(before_png)).decode()
    after_b64 = base64.b64encode(resize_to_logical_png(after_png)).decode()
    messages = [
        SystemMessage(content=BACK_PROMPT),
        HumanMessage(content=[
            {
                "type": "text",
                "text": (
                    "第一张(BEFORE)是点击前的页面，第二张(AFTER)是点击后跳转的页面。"
                    "请找出从 AFTER 返回 BEFORE 的方法。"
                ),
            },
            {"type": "text", "text": "BEFORE（点击前）:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_b64}"}},
            {"type": "text", "text": "AFTER（点击后）:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_b64}"}},
        ]),
    ]
    action = invoke_structured(llm, messages, BackAction)
    if not action.can_go_back:
        return None
    if not (0 <= action.back_x <= 1000 and 0 <= action.back_y <= 1000):
        return None
    return action


def _tap_back_once(
    client,
    screenshot: Callable[[], bytes],
    initial_page: ParsedPage,
    initial_bytes: bytes,
    before_back_bytes: bytes | None,
    tap_dir: Path,
    index: int,
    element_type: str,
    initial_fingerprint: str | None,
    attempt: int,
    llm_back_action: BackAction | None = None,
    parent_bytes: bytes | None = None,
) -> tuple[bool, bool, str | None]:
    """One attempt to go back. Returns (matched_initial, on_parent, fingerprint)."""

    if llm_back_action is not None:
        lx, ly, _ = tap_llm_back(client, llm_back_action)
        action_desc = f"[LLM] {llm_back_action.method}({lx:.0f},{ly:.0f})"
    elif attempt <= len(BACK_TAP_POINTS):
        lx, ly, _ = tap_back(client, attempt)
        action_desc = f"[{attempt}] 左上角({lx:.0f},{ly:.0f})"
    else:
        (sx, sy), (ex, ey), _ = swipe_back(client)
        action_desc = f"[{attempt}] 左滑({sx:.0f},{sy:.0f})→({ex:.0f},{ey:.0f})"
    time.sleep(BACK_SETTLE_SECONDS)

    back_bytes = screenshot()
    back_path = tap_dir / f"tap_{index:02d}_{element_type}_back_{attempt}.png"
    if back_bytes:
        back_path.write_bytes(back_bytes)

    if before_back_bytes and back_bytes:
        action_similarity = png_similarity(before_back_bytes, back_bytes)
        if action_similarity >= BACK_ACTION_NO_CHANGE_THRESHOLD:
            print(f"    ↩ {action_desc} → 未变化")
            return False, False, initial_fingerprint

    # Check parent page first (cheap pixel comparison)
    if parent_bytes and back_bytes:
        parent_sim = png_similarity(parent_bytes, back_bytes)
        if parent_sim >= BACK_MATCH_THRESHOLD:
            print(f"    ↩ {action_desc} → 父页面 {parent_sim:.3f}")
            return False, True, initial_fingerprint

    decision = _matches_initial_layered(initial_page, initial_bytes, back_bytes, initial_fingerprint)
    status = f"✓ {decision.similarity:.3f}" if decision.matched else f"✗ {decision.similarity:.3f}"
    print(f"    ↩ {action_desc} → {status}")
    return bool(decision.matched), False, initial_fingerprint


def return_to_initial(
    client,
    screenshot: Callable[[], bytes],
    initial_page: ParsedPage,
    initial_bytes: bytes,
    before_back_bytes: bytes | None,
    tap_dir: Path,
    index: int,
    element_type: str,
    initial_fingerprint: str | None = None,
    parent_bytes: bytes | None = None,
) -> tuple[bool, bool]:
    """Navigate back to the initial page, retrying up to BACK_MAX_RETRIES times.

    Returns (matched_initial, on_parent).
    """
    fp = initial_fingerprint
    clicked_page_bytes = before_back_bytes

    for attempt in range(1, BACK_MAX_RETRIES + 1):
        matched, on_parent, fp = _tap_back_once(
            client, screenshot, initial_page, initial_bytes,
            before_back_bytes, tap_dir, index, element_type, fp, attempt,
            parent_bytes=parent_bytes,
        )
        if matched:
            return True, False
        if on_parent:
            return False, True
        before_back_bytes = screenshot()

    # LLM fallback: ask vision model for back action
    llm_action = infer_back_action(initial_bytes, clicked_page_bytes)
    if llm_action is not None:
        matched, on_parent, fp = _tap_back_once(
            client, screenshot, initial_page, initial_bytes,
            before_back_bytes, tap_dir, index, element_type, fp, 0,
            llm_back_action=llm_action, parent_bytes=parent_bytes,
        )
        if matched:
            return True, False
        if on_parent:
            return False, True

    print(f"    {BACK_MAX_RETRIES} 次返回尝试后仍未回到初始页面")
    return False, False


def probe_elements(
    client,
    knowledge: PageKnowledge,
    out_dir: Path,
    initial_screenshot_path: Path | None = None,
    screenshot: Callable[[], bytes] | None = None,
    debug: bool = False,
    sample: int = 0,
    parent_bytes: bytes | None = None,
    re_nav: tuple[float, float] | None = None,
) -> ReconResult:
    """Tap each area, capture after-state, return structured result.

    Args:
        sample: If > 0, randomly sample this many elements instead of probing all.
        parent_bytes: If given, check if a tap navigated back to parent page.
        re_nav: (lx, ly) to re-navigate from parent to child page when parent detected.
    """
    import random
    from policy_expr.executor import logical_xy

    page = knowledge.page
    areas = knowledge.areas
    has_nav = page.bottom_nav.has_nav

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

    tap_dir = out_dir / "tap"
    tap_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "recon_result.json"

    for i, area in enumerate(areas, 1):
        ax, ay = area.center_xy
        lx, ly = logical_xy(ax, ay)
        is_tab = has_nav and ay > 900
        print(f"\n  [{i}/{len(areas)}] 「{area.label}」 @ ({ax:.0f},{ay:.0f}) → ({lx:.0f},{ly:.0f})")

        tap_response = client.tap(lx, ly)
        tap_ok = "failed" not in tap_response.lower() and "interrupted" not in tap_response.lower()
        print(f"    结果: {tap_response}")
        time.sleep(2.0)

        after_bytes = screenshot()
        after_path = tap_dir / f"tap_{i:02d}_{area.label}.png"
        if after_bytes:
            after_path.write_bytes(after_bytes)
            print(f"    截图: {after_path}")

        result.taps.append(TapResult(
            index=i,
            element_type="tab" if is_tab else "area",
            label=area.label,
            x=ax,
            y=ay,
            tap_ok=tap_ok,
            screenshot_path=str(after_path),
            navigated=not is_tab,
        ))

        if not is_tab:
            # Fast parent check on the immediate tap result (before any back attempts)
            on_parent = bool(
                parent_bytes and after_bytes
                and png_similarity(parent_bytes, after_bytes) >= BACK_MATCH_THRESHOLD
            )
            if not on_parent and initial_bytes:
                matched, on_parent = return_to_initial(
                    client, screenshot, page, initial_bytes,
                    after_bytes, tap_dir, i, "area",
                    parent_bytes=parent_bytes,
                )
                if not matched and not on_parent:
                    print("    返回后未回到初始界面，停止探测")
                    break

            if on_parent:
                if re_nav:
                    print(f"    落入父页面，重新进入子页面")
                    client.tap(re_nav[0], re_nav[1])
                    time.sleep(2.0)
                    continue
                break

        result.save(result_path)

        if debug:
            input("    [DEBUG] 按回车继续下一个区域...")

    # Ensure we return to initial page after probing
    if initial_bytes:
        current_bytes = screenshot()
        decision = _matches_initial_layered(page, initial_bytes, current_bytes, None)
        if not decision.matched:
            return_to_initial(
                client, screenshot, page, initial_bytes,
                current_bytes, tap_dir, 0, "final",
            )

    return result

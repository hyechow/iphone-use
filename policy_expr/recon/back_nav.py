"""Back-navigation: four-tier tap strategy with stack-based navigation assessment."""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm.structured import invoke_structured
from policy_expr.config import resolve_llm_config
from policy_expr.executor import is_valid_tap
from policy_expr.policies.base import resize_to_logical_png
from policy_expr.recon.page_compare import PageComparator, make_comparator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACK_TAP_CENTER = (70.0, 125.0)
BACK_TAP_MAX_DIST = 150.0
BACK_SETTLE_SECONDS = 1.5

# Two comparators with different responsibilities:
# - _change_comp: edge IoU — fast no_change detection inside _try_tap
# - _identity_comp: GUIClip — semantic page identity for _match_stack and page_records
# Preload GUIClip at module import time to avoid loading it during exploration
_change_comp: PageComparator = make_comparator("edge_iou")
_identity_comp: PageComparator = make_comparator("guiclip")  # Triggers GUIClip model loading


def _get_change_comp() -> PageComparator:
    return _change_comp


def _get_identity_comp() -> PageComparator:
    return _identity_comp


# ---------------------------------------------------------------------------
# LLM models
# ---------------------------------------------------------------------------

class BackAction(BaseModel):
    can_go_back: bool = Field(description="能否找到返回到前一页的方法")
    method: str = Field(description="返回方法描述")
    back_x: float = Field(default=-1, description="返回目标归一化 x 坐标（0-1000）")
    back_y: float = Field(default=-1, description="返回目标归一化 y 坐标（0-1000）")


BACK_PROMPT = """\
你是一个手机页面导航分析器。

用户给出了两张截图：
- 第一张（BEFORE）：点击前的原始页面
- 第二张（AFTER）：点击某个元素后的页面

还提供了导航上下文：用户是如何从 BEFORE 到达 AFTER 的（例如"点击了底部「发现」tab"、"点击了「珠珠」聊天项"）。

目标：找到一个点击动作，使 AFTER 页面更接近 BEFORE 页面。

处理优先级：
1. 如果 AFTER 有弹窗/对话框/广告浮层覆盖页面 → 点击关闭/取消/跳过/拒绝按钮关掉弹窗
2. 如果是通过底部 tab 切换进入的 → 点击原来的底部 tab 返回
3. 如果是正常跳转的新页面 → 点击左上角返回按钮、或关闭按钮（×）

每次只输出一个动作（先处理弹窗，弹窗没了再考虑页面导航）。

输出：
- can_go_back: 是否找到可点击的目标
- method: 描述这次点击的作用
- back_x, back_y: 目标坐标（0-1000，左上角原点）
"""


# ---------------------------------------------------------------------------
# Tap operations
# ---------------------------------------------------------------------------

def tap_back(client) -> tuple[float, float, str]:
    """Tap the iOS back button area."""
    from policy_expr.executor import logical_xy

    lx, ly = logical_xy(*BACK_TAP_CENTER)
    result = client.tap(lx, ly)
    return lx, ly, result
    result = client.tap(lx, ly)
    return lx, ly, result


def tap_llm_back(client, action: BackAction) -> tuple[float, float, str] | None:
    """Tap the return target selected by the vision model. Returns None if coords invalid."""
    from policy_expr.executor import logical_xy

    if not is_valid_tap(action.back_x, action.back_y):
        return None
    lx, ly = logical_xy(action.back_x, action.back_y)
    result = client.tap(lx, ly)
    return lx, ly, result


def tap_yolo_back(client, png_bytes: bytes) -> tuple[float, float, str] | None:
    """Tap the YOLO-detected icon nearest to the back button region."""
    from policy_expr.executor import logical_xy
    from policy_expr.recon.yolo_calibrator import YoloCalibrator

    cal = YoloCalibrator.from_png(png_bytes)
    if cal is None:
        return None
    point = cal.nearest(*BACK_TAP_CENTER, max_dist=BACK_TAP_MAX_DIST)
    if point is None:
        return None
    lx, ly = logical_xy(point[0], point[1])
    resp = client.tap(lx, ly)
    return lx, ly, resp


# ---------------------------------------------------------------------------
# LLM back action inference
# ---------------------------------------------------------------------------

def infer_back_action(before_png: bytes, after_png: bytes | None, nav_context: str = "") -> BackAction | None:
    """Ask the vision model how to navigate from AFTER back to BEFORE.

    Args:
        nav_context: How the user navigated from BEFORE to AFTER
                     (e.g. "点击了底部「发现」tab", "点击了「珠珠」聊天项").
    """
    if not after_png:
        return None

    cfg = resolve_llm_config("back_nav")
    llm = ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=0,
    )
    before_b64 = base64.b64encode(resize_to_logical_png(before_png)).decode()
    after_b64 = base64.b64encode(resize_to_logical_png(after_png)).decode()

    context_text = (
        "第一张(BEFORE)是点击前的页面，第二张(AFTER)是点击后跳转的页面。"
        "请找出从 AFTER 返回 BEFORE 的方法。"
    )
    if nav_context:
        context_text += f"\n\n导航上下文：用户通过「{nav_context}」从 BEFORE 到达了 AFTER。"

    messages = [
        SystemMessage(content=BACK_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": context_text},
            {"type": "text", "text": "BEFORE（点击前）:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_b64}"}},
            {"type": "text", "text": "AFTER（点击后）:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_b64}"}},
        ]),
    ]
    action = invoke_structured(llm, messages, BackAction)
    if not action.can_go_back:
        return None
    if not is_valid_tap(action.back_x, action.back_y):
        return None
    return action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def back_shot_path(out_dir: Path | None, tap_index: int, attempt_num: int) -> Path | None:
    if out_dir is None:
        return None
    prefix = f"tap_{tap_index:02d}" if tap_index > 0 else "end"
    return out_dir / f"{prefix}_back_{attempt_num:02d}.png"


def save_if_changed(png_bytes: bytes | None, path: Path | None) -> str:
    """Save screenshot only when the screen actually changed. Returns path str or ''."""
    if path and png_bytes:
        path.write_bytes(png_bytes)
        return str(path)
    return ""


# ---------------------------------------------------------------------------
# Core navigation
# ---------------------------------------------------------------------------

def _match_stack(
    comp: PageComparator,
    stack: list[tuple[bytes, tuple[float, float] | None]],
    current_bytes: bytes,
) -> int:
    """Match current page against the nav stack. Returns stack index or -1."""
    for i, (page_bytes, _) in enumerate(stack):
        if comp.is_same_page(page_bytes, current_bytes).matched:
            return i
    return -1


def _navigate_forward(
    client,
    stack: list[tuple[bytes, tuple[float, float] | None]],
    from_level: int,
    screenshot: Callable[[], bytes],
    log: list[dict] | None = None,
    tap_index: int = 0,
    out_dir: Path | None = None,
) -> None:
    """Navigate forward from stack[from_level] to the top of stack (initial page)."""
    for i in range(from_level, len(stack) - 1):
        coords = stack[i][1]
        if coords is None:
            continue
        client.tap(*coords)
        time.sleep(BACK_SETTLE_SECONDS)
        after = screenshot()
        if log is not None:
            shot_str = ""
            if out_dir is not None and after:
                shot_path = back_shot_path(out_dir, tap_index, len(log) + 1)
                if shot_path:
                    shot_path.write_bytes(after)
                    shot_str = str(shot_path)
            log.append({"strategy": "forward", "coords": [round(coords[0]), round(coords[1])],
                        "result": f"L{i}→L{i+1}", "success": True, "screenshot": shot_str})


def _yolo_detect(png_bytes: bytes) -> tuple[float, float] | None:
    """YOLO detect nearest back/close icon. Returns normalized (ax, ay) or None."""
    from policy_expr.recon.yolo_calibrator import YoloCalibrator
    cal = YoloCalibrator.from_png(png_bytes)
    if cal is None:
        return None
    return cal.nearest(*BACK_TAP_CENTER, max_dist=BACK_TAP_MAX_DIST)


def _try_tap(
    client,
    screenshot: Callable[[], bytes],
    before_bytes: bytes | None,
    strategy: str,
    tap_fn: Callable[[], tuple | None],
    log: list[dict],
    save_path: Path | None = None,
) -> tuple[float, float, bytes] | None:
    """Execute one tap, check if page changed.

    tap_fn: performs the tap, returns (lx, ly, ...) or None if can't tap.
    Returns (lx, ly, after_bytes) if page changed, None otherwise.
    """
    coords = tap_fn()
    if coords is None:
        return None

    lx, ly = coords[0], coords[1]
    time.sleep(BACK_SETTLE_SECONDS)
    after_bytes = screenshot()

    comp = _get_change_comp()
    if before_bytes and after_bytes:
        unchanged, score = comp.no_change_score(before_bytes, after_bytes)
        if unchanged:
            print(f"    ↩ [{strategy}] ({lx:.0f},{ly:.0f}) → 未变化")
            log.append({"strategy": strategy, "coords": [round(lx), round(ly)],
                        "result": "未变化", "score": round(score, 3),
                        "success": False, "screenshot": ""})
            return None

    shot_str = save_if_changed(after_bytes, save_path)
    print(f"    ↩ [{strategy}] ({lx:.0f},{ly:.0f}) → 已变化")
    log.append({"strategy": strategy, "coords": [round(lx), round(ly)],
                "result": "已变化", "screenshot": shot_str})
    return lx, ly, after_bytes


def _execute_tap_tiers(
    client,
    screenshot: Callable[[], bytes],
    before_bytes: bytes | None,
    initial_bytes: bytes,
    nav_context: str,
    out_dir: Path | None,
    tap_index: int,
    log: list[dict],
    skip_mechanical: bool = False,
) -> tuple[float, float, bytes] | None:
    """Try Tier 1→2→3→4 until one produces a page change.

    Each tier only cares: did the tap work (page changed)?
    Navigation assessment is handled by the caller.
    Tiers group into two families:
      mechanical (Tier 1+2): fixed position + YOLO calibration — same back-button region
      llm (Tier 3+4): LLM inference + optional YOLO position correction
    skip_mechanical: skip the mechanical family entirely, go straight to LLM.
    """
    from policy_expr.executor import logical_xy

    yolo_point = None
    lx, ly = 0.0, 0.0
    # current_bytes is used for LLM; updated after each tap attempt
    # original_bytes is the screenshot before any tapping, used for YOLO calibration
    current_bytes: bytes = before_bytes or initial_bytes
    original_bytes: bytes = current_bytes  # Keep original for YOLO calibration

    if not skip_mechanical:
        # ── Tier 1: Fixed-position back tap ──
        save_path = back_shot_path(out_dir, tap_index, len(log) + 1)
        result = _try_tap(
            client, screenshot, before_bytes, "fixed",
            lambda: tap_back(client), log, save_path,
        )
        if result is not None:
            return result

        # ── Tier 2: YOLO calibration (only when Tier 1 fails) ──
        current_bytes = screenshot()
        yolo_point = _yolo_detect(current_bytes)
        if yolo_point is not None:
            lx, ly = logical_xy(*yolo_point)
            save_path = back_shot_path(out_dir, tap_index, len(log) + 1)
            result = _try_tap(
                client, screenshot, before_bytes, "YOLO",
                lambda: (client.tap(lx, ly), (lx, ly))[1], log, save_path,
            )
            if result is not None:
                return result
        else:
            log.append({"strategy": "YOLO", "result": "搜索范围内无图标",
                        "success": False, "screenshot": ""})

    # ── Tier 3: LLM inference ──
    llm_action = infer_back_action(initial_bytes, current_bytes, nav_context=nav_context)
    if llm_action is None:
        print("    [LLM] 未能识别返回动作")
        log.append({"strategy": "LLM", "result": "未能识别返回动作",
                    "success": False, "screenshot": ""})
        return None

    llm_result = tap_llm_back(client, llm_action)
    if llm_result is not None:
        save_path = back_shot_path(out_dir, tap_index, len(log) + 1)
        result = _try_tap(
            client, screenshot, before_bytes, "LLM",
            lambda: llm_result, log, save_path,
        )
        if result is not None:
            return result
    else:
        bx, by = llm_action.back_x, llm_action.back_y
        print(f"    [LLM] {llm_action.method}({bx:.0f},{by:.0f}) → 无效坐标")
        log.append({"strategy": "LLM", "coords": [round(bx), round(by)],
                    "result": "未变化", "success": False, "screenshot": ""})

    # ── Tier 4: LLM + YOLO ──
    # Calibrate LLM output by finding nearest icon to LLM coordinates
    if llm_action is not None:
        from policy_expr.recon.yolo_calibrator import YoloCalibrator
        # Use original_bytes for YOLO calibration to ensure we're on the right page
        cal = YoloCalibrator.from_png(original_bytes)
        if cal is not None:
            calibrated_point = cal.nearest(llm_action.back_x, llm_action.back_y, max_dist=100.0)
            if calibrated_point is not None:
                cx, cy = calibrated_point
                lx, ly = logical_xy(cx, cy)
                save_path = back_shot_path(out_dir, tap_index, len(log) + 1)
                result = _try_tap(
                    client, screenshot, before_bytes, "LLM+YOLO",
                    lambda: (client.tap(lx, ly), (lx, ly))[1], log, save_path,
                )
                if result is not None:
                    return result
            else:
                print(f"    [LLM+YOLO] LLM坐标({llm_action.back_x:.0f},{llm_action.back_y:.0f})附近无YOLO检测结果")
                log.append({"strategy": "LLM+YOLO",
                            "result": f"LLM坐标附近无检测结果",
                            "success": False, "screenshot": ""})
        else:
            print("    [LLM+YOLO] YOLO检测失败")
            log.append({"strategy": "LLM+YOLO", "result": "YOLO检测失败",
                        "success": False, "screenshot": ""})
    else:
        print("    [LLM+YOLO] 无LLM输出可供校正")
        log.append({"strategy": "LLM+YOLO", "result": "无LLM输出",
                    "success": False, "screenshot": ""})

    return None


def return_to_initial(
    client,
    screenshot: Callable[[], bytes],
    nav_stack: list[tuple[bytes, tuple[float, float] | None]],
    before_back_bytes: bytes | None = None,
    out_dir: Path | None = None,
    tap_index: int = 0,
    nav_context: str = "",
    page_records: list[tuple[bytes, set[str]]] | None = None,
) -> tuple[bool, list[dict]]:
    """Navigate back to the initial page (top of nav_stack).

    Architecture: two-phase loop.
      Phase 1 (_execute_tap_tiers): try tier families until one produces a page change.
      Phase 2 (this function): assess "where are we now?" via stack matching.
      Unknown page → track per-page tier history, loop back to Phase 1.

    page_records maps each visited page (identified via GUIClip) to the set of tier
    families already tried from it: "mechanical" (Tier 1+2) and/or "llm" (Tier 3+4).
    When both families are exhausted for the current page, we give up.

    Pass page_records from previous calls to preserve strategy memory across
    multiple return_to_initial invocations (e.g., when exploring child pages in DFS).
    """
    log: list[dict] = []
    id_comp = _get_identity_comp()
    top_level = len(nav_stack) - 1
    initial_bytes = nav_stack[top_level][0]
    max_rounds = 6

    # Per-page tier attempt history: list of (page_screenshot, tried_families).
    # GUIClip is used for page matching so structurally-similar but content-different
    # pages (e.g. 个人资料 vs 我的页面) are correctly distinguished.
    # If page_records is provided, reuse it to preserve strategy memory across calls.
    if page_records is None:
        page_records = []

    def _get_tried(current_bytes: bytes) -> set[str]:
        """Return the tried-families set for current page, creating entry if new."""
        for page_bytes, tried in page_records:
            if id_comp.is_same_page(page_bytes, current_bytes).matched:
                return tried
        tried: set[str] = set()
        page_records.append((current_bytes, tried))
        return tried

    current_bytes = before_back_bytes or initial_bytes

    for _ in range(max_rounds):
        tried = _get_tried(current_bytes)

        if "mechanical" in tried and "llm" in tried:
            print("    [nav] 当前页面所有策略均已尝试，放弃")
            break

        skip_mechanical = "mechanical" in tried
        if skip_mechanical:
            print("    [nav] 当前页面已试过机械回退，直接调用 LLM")

        prev_log_len = len(log)

        # Phase 1: try tier families until one produces a page change
        tap_result = _execute_tap_tiers(
            client, screenshot, current_bytes, initial_bytes,
            nav_context, out_dir, tap_index, log,
            skip_mechanical=skip_mechanical,
        )

        # Record which families were executed in this call
        for entry in log[prev_log_len:]:
            s = entry.get("strategy", "")
            if s in ("fixed", "YOLO"):
                tried.add("mechanical")
            elif s in ("LLM", "LLM+YOLO"):
                tried.add("llm")

        if tap_result is None:
            break  # all remaining tiers for this page exhausted

        _, _, back_bytes = tap_result

        # Phase 2: assess where we ended up
        matched_level = _match_stack(id_comp, nav_stack, back_bytes)
        if matched_level >= 0:
            is_initial = matched_level == top_level
            sim = id_comp.raw_similarity(nav_stack[matched_level][0], back_bytes)
            level_desc = "initial" if is_initial else f"L{matched_level}"
            print(f"    → 匹配 {level_desc} ({sim:.3f})")
            if log:
                log[-1]["result"] = level_desc
                log[-1]["score"] = round(sim, 3)
                log[-1]["success"] = is_initial
            if not is_initial:
                print(f"    匹配到 L{matched_level}，forward 回 initial")
                _navigate_forward(client, nav_stack, matched_level, screenshot,
                                  log=log, tap_index=tap_index, out_dir=out_dir)
            return True, log

        # Unknown page — check if it's a parallel page
        sim_to_initial = id_comp.raw_similarity(initial_bytes, back_bytes)

        # Calculate max similarity with all nav_stack pages
        max_sim = 0.0
        for page_bytes, _ in nav_stack:
            sim = id_comp.raw_similarity(page_bytes, back_bytes)
            max_sim = max(max_sim, sim)

        print(f"    → 未知页 (与initial: {sim_to_initial:.3f}, 与stack最大: {max_sim:.3f})")
        if log:
            log[-1]["result"] = f"未知页 (initial: {sim_to_initial:.3f}, stack_max: {max_sim:.3f})"
            log[-1]["score"] = round(sim_to_initial, 3)

        # Parallel page detection: if low similarity with all stack pages, tap bottom tab
        if max_sim < 0.75:
            print(f"    [平行页] 检测为平行页面 (最大相似度 {max_sim:.3f} < 0.75)")
            # Try tapping bottom tab to return to common ancestor
            from policy_expr.executor import logical_xy
            tap_x, tap_y = logical_xy(500.0, 950.0)  # Approximate bottom tab center
            client.tap(tap_x, tap_y)
            time.sleep(BACK_SETTLE_SECONDS)
            after_tab_bytes = screenshot()

            # Check if tapping bottom tab helped
            tab_match = _match_stack(id_comp, nav_stack, after_tab_bytes)
            if tab_match >= 0:
                print(f"    [平行页] bottom_tab后匹配到 L{tab_match}")
                if log:
                    log.append({"strategy": "bottom_tab", "coords": [round(tap_x), round(tap_y)],
                                "result": f"L{tab_match}", "success": True, "screenshot": ""})
                current_bytes = after_tab_bytes
                continue  # Try normal back navigation from this new page
            else:
                print(f"    [平行页] bottom_tab后仍未匹配到stack")
                if log:
                    log.append({"strategy": "bottom_tab", "coords": [round(tap_x), round(tap_y)],
                                "result": "未知页", "success": False, "screenshot": ""})

        current_bytes = back_bytes

    print("    所有回退策略均未成功返回初始页面")
    return False, log


def manual_recover(
    client,
    screenshot,
    nav_stack: list[tuple[bytes, tuple[float, float] | None]],
    top_level: int,
    prompt: str = "",
    max_attempts: int = 3,
) -> bool:
    """Prompt user to manually navigate back to initial page.

    Saves the target page screenshot to /tmp for user reference.
    Returns True if the user successfully recovered, False if aborted.
    """
    import tempfile
    comp = _get_identity_comp()
    initial_bytes = nav_stack[top_level][0]

    ref_path = Path(tempfile.gettempdir()) / "recon_target_page.png"
    ref_path.write_bytes(initial_bytes)

    for attempt in range(1, max_attempts + 1):
        print(f"\n  ⚠ {prompt}")
        print(f"  目标截图: {ref_path}")
        print(f"  请手动操作手机回到目标页面，然后按回车继续 ({attempt}/{max_attempts})")
        print(f"  （输入 q 放弃当前页面的探测）", end="", flush=True)
        user_input = input()

        if user_input.strip().lower() == "q":
            print("  用户放弃，终止探测")
            return False

        current_bytes = screenshot()
        sim = comp.raw_similarity(initial_bytes, current_bytes)
        print(f"  当前截图与初始页相似度: {sim:.3f}")

        if sim >= comp._no_change_threshold:
            print(f"  ✓ 已回到初始页")
            return True

        matched_level = _match_stack(comp, nav_stack, current_bytes)
        if matched_level >= 0:
            if matched_level == top_level:
                print(f"  ✓ 已回到初始页")
                return True
            print(f"  匹配到祖先页 L{matched_level}，尝试自动 forward 回初始页...")
            _navigate_forward(client, nav_stack, matched_level, screenshot)
            verify_bytes = screenshot()
            verify_sim = comp.raw_similarity(initial_bytes, verify_bytes)
            if verify_sim >= comp._no_change_threshold:
                print(f"  ✓ forward 成功，已回到初始页 ({verify_sim:.3f})")
                return True
            print(f"  forward 后仍未到达初始页 ({verify_sim:.3f})")

    print("  多次尝试未能回到初始页，终止探测")
    return False

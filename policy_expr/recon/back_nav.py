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
BACK_TAP_MAX_DIST = 80.0
BACK_SETTLE_SECONDS = 1.5

# Module-level comparator (lazy, edge IoU by default)
_comparator: PageComparator | None = None


def _get_comparator() -> PageComparator:
    global _comparator
    if _comparator is None:
        _comparator = make_comparator()
    return _comparator


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
    if not (0 <= action.back_x <= 1000 and 0 <= action.back_y <= 1000):
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

    comp = _get_comparator()
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
    skip_yolo: bool = False,
) -> tuple[float, float, bytes] | None:
    """Try Tier 1→2→3→4 until one produces a page change.

    Each tier only cares: did the tap work (page changed)?
    Navigation assessment is handled by the caller.
    skip_yolo: skip YOLO-based tiers (2, 4) when primary tiers (1, 3) already work.
    """
    from policy_expr.executor import logical_xy

    # ── Tier 1: Fixed-position back tap ──
    save_path = back_shot_path(out_dir, tap_index, len(log) + 1)
    result = _try_tap(
        client, screenshot, before_bytes, "fixed",
        lambda: tap_back(client), log, save_path,
    )
    if result is not None:
        return result

    # ── Tier 2: YOLO icon detection (skip if primary tiers work) ──
    yolo_point = None
    lx, ly = 0.0, 0.0
    current_bytes = screenshot()
    if not skip_yolo:
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
            print("    [YOLO] 搜索范围内无图标")
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

        # ── Tier 4: LLM + YOLO (skip if primary tiers work) ──
        if not skip_yolo and yolo_point is not None:
            save_path = back_shot_path(out_dir, tap_index, len(log) + 1)
            result = _try_tap(
                client, screenshot, before_bytes, "LLM+YOLO",
                lambda: (client.tap(lx, ly), (lx, ly))[1], log, save_path,
            )
            if result is not None:
                return result
        elif not skip_yolo:
            print("    [LLM+YOLO] 无YOLO检测结果可用于校正")
            log.append({"strategy": "LLM+YOLO", "result": "无YOLO检测结果",
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
) -> tuple[bool, list[dict]]:
    """Navigate back to the initial page (top of nav_stack).

    Architecture: two-phase loop.
      Phase 1 (_execute_tap_tiers): try four tap strategies, only care "did tap work?"
      Phase 2 (this function): assess "where are we now?" via stack matching.
      If unknown page → loop back to Phase 1.
    """
    log: list[dict] = []
    comp = _get_comparator()
    top_level = len(nav_stack) - 1
    initial_bytes = nav_stack[top_level][0]
    max_rounds = 4
    primary_worked: set[str] = set()  # track which primary tiers ("fixed", "LLM") produced changes

    for _ in range(max_rounds):
        # Skip YOLO tiers if both primary tiers already proved they can tap
        skip_yolo = "fixed" in primary_worked and "LLM" in primary_worked

        # Phase 1: try tap tiers until one produces a page change
        tap_result = _execute_tap_tiers(
            client, screenshot, before_back_bytes, initial_bytes,
            nav_context, out_dir, tap_index, log,
            skip_yolo=skip_yolo,
        )
        if tap_result is None:
            break  # all tiers exhausted

        _, _, back_bytes = tap_result

        # Phase 2: assess where we ended up
        matched_level = _match_stack(comp, nav_stack, back_bytes)
        if matched_level >= 0:
            is_initial = matched_level == top_level
            sim = comp.raw_similarity(nav_stack[matched_level][0], back_bytes)
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

        # Unknown page → record primary tier, update before_bytes and loop back
        sim_to_initial = comp.raw_similarity(initial_bytes, back_bytes)
        print(f"    → 未知页 ({sim_to_initial:.3f})")
        if log:
            strategy = log[-1].get("strategy", "")
            if strategy.startswith("fixed"):
                primary_worked.add("fixed")
            elif strategy in ("LLM", "LLM+YOLO"):
                primary_worked.add("LLM")
            log[-1]["result"] = f"未知页 ({sim_to_initial:.3f})"
            log[-1]["score"] = round(sim_to_initial, 3)
        before_back_bytes = back_bytes

    print("    所有回退策略均未成功返回初始页面")
    return False, log

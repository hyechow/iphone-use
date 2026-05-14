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
from policy_expr.perception import try_resume_mac
from policy_expr.recon.page_parser import PageKnowledge
from policy_expr.recon.utils import (
    ProbeAbortedError,
    ReconResult,
    ScreenMatchDecision,
    TapResult,
    png_similarity,
)


BACK_TAP_CENTER = (70.0, 125.0)
BACK_TAP_MAX_DIST = 80.0
BACK_TAP_JITTER = 10.0
BACK_SETTLE_SECONDS = 1.5
BACK_ACTION_NO_CHANGE_THRESHOLD = 0.80
BACK_MATCH_THRESHOLD = 0.20
BACK_TAP_POINTS = (BACK_TAP_CENTER,)
RECOVERY_MAX_ATTEMPTS = 3


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

目标：找到一个点击动作，使 AFTER 页面更接近 BEFORE 页面。

处理优先级：
1. 如果 AFTER 有弹窗/对话框/广告浮层覆盖页面 → 点击关闭/取消/跳过/拒绝按钮关掉弹窗
2. 如果 AFTER 是正常跳转的新页面 → 点击左上角返回按钮、底部 tab、或关闭按钮（×）

每次只输出一个动作（先处理弹窗，弹窗没了再考虑页面导航）。

输出：
- can_go_back: 是否找到可点击的目标
- method: 描述这次点击的作用
- back_x, back_y: 目标坐标（0-1000，左上角原点）
"""



def _matches_initial_layered(
    initial_png: bytes,
    current_png: bytes | None,
) -> ScreenMatchDecision:
    if not current_png:
        return ScreenMatchDecision(False, 0.0, "screenshot", "missing current screenshot")

    similarity = png_similarity(initial_png, current_png)
    matched = similarity >= BACK_MATCH_THRESHOLD
    return ScreenMatchDecision(
        matched,
        similarity,
        "pixel",
        f"similarity {similarity:.3f} {'above' if matched else 'below'} back match threshold {BACK_MATCH_THRESHOLD}",
    )


def _same_page_fingerprint(png1: bytes, png2: bytes) -> bool:
    """Check if two screenshots belong to the same structural page via fingerprint."""
    from policy_expr.recon.fingerprint import compute_fingerprint
    return compute_fingerprint(png1).key == compute_fingerprint(png2).key



def _back_tap_point(attempt: int) -> tuple[float, float]:
    point = BACK_TAP_POINTS[(attempt - 1) % len(BACK_TAP_POINTS)]
    if attempt <= len(BACK_TAP_POINTS):
        return point
    angle = np.random.uniform(0, 2 * np.pi)
    r = BACK_TAP_JITTER * np.sqrt(np.random.uniform(0, 1))
    return (point[0] + r * np.cos(angle), point[1] + r * np.sin(angle))


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


BACK_MAX_RETRIES = len(BACK_TAP_POINTS)
LLM_BACK_MAX_RETRIES = 1



def infer_back_action(before_png: bytes, after_png: bytes | None) -> BackAction | None:
    """Ask the vision model how to navigate from AFTER back to BEFORE."""
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


def _back_shot_path(out_dir: Path | None, tap_index: int, attempt_num: int) -> Path | None:
    if out_dir is None:
        return None
    prefix = f"tap_{tap_index:02d}" if tap_index > 0 else "end"
    return out_dir / f"{prefix}_back_{attempt_num:02d}.png"


def _save_if_changed(png_bytes: bytes | None, path: Path | None) -> str:
    """Save screenshot only when the screen actually changed. Returns path str or ''."""
    if path and png_bytes:
        path.write_bytes(png_bytes)
        return str(path)
    return ""


def _tap_back_once(
    client,
    screenshot: Callable[[], bytes],
    initial_bytes: bytes,
    before_back_bytes: bytes | None,
    attempt: int,
    llm_back_action: BackAction | None = None,
    parent_bytes: bytes | None = None,
    log: list[dict] | None = None,
    save_path: Path | None = None,
) -> tuple[bool, bool, bool]:
    """One attempt to go back.

    Returns (matched_initial, on_parent, page_changed).
    page_changed=True means the tap navigated somewhere (not just "unchanged").
    """

    if llm_back_action is not None:
        lx, ly, _ = tap_llm_back(client, llm_back_action)
        strategy = "LLM"
        action_desc = f"[LLM] {llm_back_action.method}({lx:.0f},{ly:.0f})"
    else:
        lx, ly, _ = tap_back(client, attempt)
        strategy = f"fixed_{attempt}"
        action_desc = f"[{attempt}] 左上角({lx:.0f},{ly:.0f})"
    time.sleep(BACK_SETTLE_SECONDS)

    back_bytes = screenshot()

    # Check parent page first (cheap pixel comparison)
    if parent_bytes and back_bytes:
        parent_sim = png_similarity(parent_bytes, back_bytes)
        if parent_sim >= BACK_MATCH_THRESHOLD:
            print(f"    ↩ {action_desc} → 父页面 {parent_sim:.3f}")
            shot_str = _save_if_changed(back_bytes, save_path)
            if log is not None:
                log.append({"strategy": strategy, "coords": [round(lx), round(ly)],
                            "result": "父页面", "score": round(parent_sim, 3), "success": False,
                            "screenshot": shot_str})
            return False, True, True

    if before_back_bytes and back_bytes:
        action_similarity = png_similarity(before_back_bytes, back_bytes)
        if action_similarity >= BACK_ACTION_NO_CHANGE_THRESHOLD:
            print(f"    ↩ {action_desc} → 未变化")
            if log is not None:
                log.append({"strategy": strategy, "coords": [round(lx), round(ly)],
                            "result": "未变化", "score": round(action_similarity, 3), "success": False,
                            "screenshot": ""})  # no save: screen unchanged
            return False, False, False

    decision = _matches_initial_layered(initial_bytes, back_bytes)
    status = f"✓ {decision.similarity:.3f}" if decision.matched else f"✗ {decision.similarity:.3f}"
    print(f"    ↩ {action_desc} → {status}")
    shot_str = _save_if_changed(back_bytes, save_path)
    if log is not None:
        log.append({"strategy": strategy, "coords": [round(lx), round(ly)],
                    "result": "匹配" if decision.matched else "不匹配",
                    "score": round(decision.similarity, 3), "success": bool(decision.matched),
                    "screenshot": shot_str})
    return bool(decision.matched), False, True


def _recover_to_page(
    client,
    screenshot: Callable[[], bytes],
    target_bytes: bytes,
    log: list[dict] | None = None,
    out_dir: Path | None = None,
    tap_index: int = 0,
) -> bool:
    """Try to navigate back to a target page using back button taps.

    Handles recursive misnavigation: if a recovery tap accidentally goes
    to another sub-page, keep trying. Returns True if recovered.
    """
    for _ in range(RECOVERY_MAX_ATTEMPTS):
        current = screenshot()
        if png_similarity(current, target_bytes) >= BACK_MATCH_THRESHOLD:
            return True

        pre_bytes = current
        from policy_expr.executor import logical_xy
        lx, ly = logical_xy(*BACK_TAP_CENTER)
        client.tap(lx, ly)
        time.sleep(BACK_SETTLE_SECONDS)

        after_bytes = screenshot()
        save_path = _back_shot_path(out_dir, tap_index, len(log) + 1 if log else 1)

        if png_similarity(pre_bytes, after_bytes) >= BACK_ACTION_NO_CHANGE_THRESHOLD:
            print(f"    ↩ [recover] 未变化")
            if log is not None:
                log.append({"strategy": "recover", "coords": [round(lx), round(ly)],
                            "result": "未变化", "success": False, "screenshot": ""})
            continue

        sim_to_target = png_similarity(after_bytes, target_bytes)
        print(f"    ↩ [recover] ({lx:.0f},{ly:.0f}) → vs原页面 {sim_to_target:.3f}")
        if log is not None:
            log.append({"strategy": "recover", "coords": [round(lx), round(ly)],
                        "result": f"恢复中 vs原页面={sim_to_target:.3f}",
                        "score": round(sim_to_target, 3), "success": False,
                        "screenshot": _save_if_changed(after_bytes, save_path)})

        if sim_to_target >= BACK_MATCH_THRESHOLD:
            return True

    return False


def return_to_initial(
    client,
    screenshot: Callable[[], bytes],
    initial_bytes: bytes,
    before_back_bytes: bytes | None,
    parent_bytes: bytes | None = None,
    out_dir: Path | None = None,
    tap_index: int = 0,
) -> tuple[bool, bool, list[dict]]:
    """Navigate back to the initial page.

    Returns (matched_initial, on_parent, attempt_log).

    Three tiers: fixed position → YOLO → LLM.
    If any tier accidentally navigates to a sub-page (page changed but not
    initial/parent), recover to the pre-tier page before trying the next tier.
    This ensures each tier operates from the same starting page.
    """
    log: list[dict] = []
    round_start_bytes = screenshot()

    # --- Tier 1: fixed-position back button taps ---
    went_deeper = False
    for attempt in range(1, BACK_MAX_RETRIES + 1):
        save_path = _back_shot_path(out_dir, tap_index, len(log) + 1)
        matched, on_parent, page_changed = _tap_back_once(
            client, screenshot, initial_bytes,
            before_back_bytes, attempt,
            parent_bytes=parent_bytes, log=log, save_path=save_path,
        )
        if matched:
            return True, False, log
        if on_parent:
            return False, True, log
        if page_changed:
            # Page changed — check direction: closer to initial = progress, farther = went deeper
            current_bytes = screenshot()
            current_to_initial = png_similarity(current_bytes, initial_bytes)
            start_to_initial = png_similarity(round_start_bytes, initial_bytes)
            if current_to_initial > start_to_initial:
                # Progress toward initial — keep trying from new page
                before_back_bytes = current_bytes
                continue
            # Went deeper — recover to pre-tier page before trying YOLO/LLM
            if _recover_to_page(client, screenshot, round_start_bytes,
                                log=log, out_dir=out_dir, tap_index=tap_index):
                went_deeper = False
            else:
                went_deeper = True
            break
        before_back_bytes = screenshot()

    # --- Tier 2: YOLO fallback ---
    if not went_deeper:
        from policy_expr.recon.yolo_calibrator import YoloCalibrator

        pre_yolo_bytes = screenshot()
        yolo_cal = YoloCalibrator.from_png(pre_yolo_bytes)

        if yolo_cal is not None:
            point = yolo_cal.nearest(*BACK_TAP_CENTER, max_dist=BACK_TAP_MAX_DIST)
            if point is not None:
                from policy_expr.executor import logical_xy
                lx, ly = logical_xy(point[0], point[1])
                action_desc = f"[YOLO] ({lx:.0f},{ly:.0f})"
                client.tap(lx, ly)
                time.sleep(BACK_SETTLE_SECONDS)

                back_bytes = screenshot()
                yolo_save = _back_shot_path(out_dir, tap_index, len(log) + 1)

                if not (before_back_bytes and back_bytes
                        and png_similarity(before_back_bytes, back_bytes) >= BACK_ACTION_NO_CHANGE_THRESHOLD):

                    if parent_bytes and back_bytes:
                        parent_sim = png_similarity(parent_bytes, back_bytes)
                        if parent_sim >= BACK_MATCH_THRESHOLD:
                            print(f"    ↩ {action_desc} → 父页面 {parent_sim:.3f}")
                            log.append({"strategy": "YOLO", "coords": [round(lx), round(ly)],
                                        "result": "父页面", "score": round(parent_sim, 3), "success": False,
                                        "screenshot": _save_if_changed(back_bytes, yolo_save)})
                            return False, True, log

                    decision = _matches_initial_layered(initial_bytes, back_bytes)
                    status = f"✓ {decision.similarity:.3f}" if decision.matched else f"✗ {decision.similarity:.3f}"
                    print(f"    ↩ {action_desc} → {status}")
                    log.append({"strategy": "YOLO", "coords": [round(lx), round(ly)],
                                "result": "匹配" if decision.matched else "不匹配",
                                "score": round(decision.similarity, 3), "success": bool(decision.matched),
                                "screenshot": _save_if_changed(back_bytes, yolo_save)})
                    if decision.matched:
                        return True, False, log
                    # Check direction: progress toward initial or went deeper
                    yolo_to_initial = png_similarity(back_bytes, initial_bytes)
                    pre_to_initial = png_similarity(pre_yolo_bytes, initial_bytes)
                    if yolo_to_initial > pre_to_initial:
                        # Progress — keep going from new page (skip recovery)
                        before_back_bytes = screenshot()
                    elif not _recover_to_page(client, screenshot, pre_yolo_bytes,
                                              log=log, out_dir=out_dir, tap_index=tap_index):
                        went_deeper = True
                else:
                    log.append({"strategy": "YOLO", "coords": [round(lx), round(ly)],
                                "result": "未变化", "success": False, "screenshot": ""})
            else:
                print("    [YOLO] 搜索范围内无图标")
                log.append({"strategy": "YOLO", "result": "搜索范围内无图标", "success": False, "screenshot": ""})
        else:
            print("    [YOLO] 未检测到图标")
            log.append({"strategy": "YOLO", "result": "未检测到图标", "success": False, "screenshot": ""})

    # --- Tier 3: LLM fallback ---
    if not went_deeper:
        current_bytes = screenshot()
        llm_action = infer_back_action(initial_bytes, current_bytes)
        if llm_action is not None:
            save_path = _back_shot_path(out_dir, tap_index, len(log) + 1)
            matched, on_parent, page_changed = _tap_back_once(
                client, screenshot, initial_bytes,
                before_back_bytes, 0,
                llm_back_action=llm_action, parent_bytes=parent_bytes, log=log, save_path=save_path,
            )
            if matched:
                return True, False, log
            if on_parent:
                return False, True, log
            if not page_changed:
                before_back_bytes = screenshot()
        else:
            print("    [LLM] 未能识别返回动作")
            log.append({"strategy": "LLM", "result": "未能识别返回动作", "success": False, "screenshot": ""})

    print("    所有回退策略均未成功返回初始页面")
    return False, False, log


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
        if after_bytes and initial_bytes:
            sim = png_similarity(initial_bytes, after_bytes)
            if sim >= BACK_ACTION_NO_CHANGE_THRESHOLD:
                print(f"    页面未变化 (相似度 {sim:.3f})，跳过")
            elif sim >= BACK_MATCH_THRESHOLD and _same_page_fingerprint(initial_bytes, after_bytes):
                print(f"    页面内容变化但未导航 (相似度 {sim:.3f}, fingerprint 一致)")
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

        if navigated:
            # Fast parent check on the immediate tap result (before any back attempts)
            on_parent = bool(
                parent_bytes and after_bytes
                and png_similarity(parent_bytes, after_bytes) >= BACK_MATCH_THRESHOLD
            )
            if not on_parent and initial_bytes:
                matched, on_parent, back_log = return_to_initial(
                    client, screenshot, initial_bytes,
                    after_bytes,
                    parent_bytes=parent_bytes,
                    out_dir=tap_dir,
                    tap_index=i,
                )
                result.taps[-1].back_attempts = back_log
                if not matched and not on_parent:
                    raise ProbeAbortedError(
                        f"无法返回子页面（探测第 {i} 个元素「{area.label}」后），终止探测",
                        failed_tap=i,
                        failed_element=area.label,
                        back_attempts=back_log,
                    )

            if on_parent:
                if re_nav:
                    print(f"    落入父页面，重新进入子页面")
                    renav_num = len(result.taps[-1].back_attempts) + 1
                    renav_path = _back_shot_path(tap_dir, i, renav_num)
                    client.tap(re_nav[0], re_nav[1])
                    time.sleep(2.0)
                    renav_bytes = screenshot()
                    renav_shot = _save_if_changed(renav_bytes, renav_path)
                    result.taps[-1].back_attempts.append({
                        "strategy": "re_nav",
                        "coords": [round(re_nav[0]), round(re_nav[1])],
                        "result": "返回子页面",
                        "score": 0.0,
                        "success": True,
                        "screenshot": renav_shot,
                    })
                    continue
                break

        result.save(result_path)

        if debug:
            input("    [DEBUG] 按回车继续下一个区域...")

    # Ensure we return to initial page after probing
    if initial_bytes:
        current_bytes = screenshot()
        decision = _matches_initial_layered(initial_bytes, current_bytes)
        if not decision.matched:
            matched, on_parent, back_log = return_to_initial(
                client, screenshot, initial_bytes,
                current_bytes,
                parent_bytes=parent_bytes,
                out_dir=tap_dir,
                tap_index=0,
            )
            if matched:
                pass
            elif on_parent and re_nav:
                client.tap(re_nav[0], re_nav[1])
                time.sleep(2.0)
            else:
                raise ProbeAbortedError(
                    "探测完成后无法返回初始页面，终止",
                    failed_tap=-1,
                    failed_element="",
                    back_attempts=back_log,
                )

    return result

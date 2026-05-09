"""Fixed action execution layer for policy experiments."""

import time

from agent.utils import paste_text
from Quartz import (
    CGEventCreateMouseEvent,
    CGEventCreateScrollWheelEvent,
    CGEventPost,
    CGEventSetLocation,
    CGWindowListCopyWindowInfo,
    kCGScrollEventUnitPixel,
    kCGHIDEventTap,
    kCGEventMouseMoved,
    kCGMouseButtonLeft,
    kCGWindowListOptionAll,
    kCGNullWindowID,
)

from policy_expr.perception import LivePhoneSession
from policy_expr.schemas import Action, ActionDecision

WIN_W = 318
WIN_H = 701
WIN_H_TAP_MAX = WIN_H - 20   # 避开底部 Home 指示条安全区（约 20px）
SCROLL_TICKS = 3
SCROLL_DELTA = 120            # pixels per tick
SCROLL_INTERVAL = 0.1         # seconds between ticks (避免 macOS 合并事件)


class ActionExecutor:
    """Execute normalized policy actions via mirroir-mcp."""

    def __init__(self, phone: LivePhoneSession):
        self.phone = phone

    def execute(self, decision: ActionDecision, app_name: str = "") -> bool:
        action = decision.action
        print(f"\n动作: [{action.action_type}] {action.description}")

        if action.action_type in ("tap", "click") and action.x is not None and action.y is not None:
            lx, ly = logical_xy(action.x, action.y)
            if not self._tap(lx, ly, decision, app_name):
                return False

        elif action.action_type == "type" and action.text:
            if action.x is not None and action.y is not None:
                lx, ly = logical_xy(action.x, action.y)
                if not self._tap(lx, ly, decision, app_name):
                    return False
                time.sleep(0.5)
            else:
                print("未提供输入坐标，默认当前输入框已聚焦，直接输入文字")
            print(f"输入文字: {action.text!r}")
            paste_text(action.text)
            print("结果: 已通过剪贴板粘贴输入")

        elif action.action_type == "scroll" and action.direction:
            self._scroll(action)

        elif action.action_type == "home":
            print("执行返回主屏")
            result = self._client().press_home()
            if "Failed to press Home" in result:
                print(f"结果: {result}，改为点击底部 Home 指示条")
                result = self._client().tap(WIN_W / 2, WIN_H - 16)
            print(f"结果: {result}")

        elif action.action_type == "stop":
            print("停止操作（当前状态已满足目标，无需执行）")
            return True

        else:
            print(f"跳过执行：action_type={action.action_type!r}，需手动处理")
            return False

        return True

    def _tap(self, lx: float, ly: float, decision: ActionDecision, app_name: str = "") -> bool:
        action = decision.action
        in_wechat = app_name.strip() in ("微信", "WeChat")
        in_bottom_right = bool(action.x and action.y and action.x > 700 and action.y > 800)
        if in_wechat and in_bottom_right:
            print("检测到微信右下角点击，等待浮层消失...")
            time.sleep(2.0)
        print(f"执行点击: ({lx:.0f}, {ly:.0f})")
        result = self._client().tap(lx, ly)
        print(f"结果: {result}")
        if "interrupted" in result.lower() or "failed" in result.lower():
            print("点击失败：落点在窗口外或操作被中断，跳过")
            return False
        return True

    def _scroll(self, action: Action) -> None:
        direction = (action.direction or "").strip().lower()
        delta = _scroll_delta(direction)

        # Find iPhone Mirroring window on screen
        origin = _find_iphone_window()
        if origin is None:
            print("  iPhone Mirroring 窗口未找到，无法发送滚轮事件")
            return
        wx, wy = origin

        # Normalized (0-1000) → screen coordinates
        ax = action.x if action.x is not None else 500
        ay = action.y if action.y is not None else 500
        sx = wx + ax / 1000 * WIN_W
        sy = wy + ay / 1000 * WIN_H

        print(f"  鼠标移至 ({sx:.0f}, {sy:.0f})，滚轮 {direction} × {SCROLL_TICKS}")

        # Move cursor to target position
        move = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (sx, sy), kCGMouseButtonLeft)
        CGEventPost(kCGHIDEventTap, move)
        time.sleep(0.1)

        # Send scroll wheel ticks
        for i in range(SCROLL_TICKS):
            ev = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitPixel, 1, delta)
            CGEventSetLocation(ev, (sx, sy))
            CGEventPost(kCGHIDEventTap, ev)
            if i < SCROLL_TICKS - 1:
                time.sleep(SCROLL_INTERVAL)

    def _client(self):
        if not self.phone.client:
            raise RuntimeError("MCP 尚未连接")
        return self.phone.client


def logical_xy(ax: float, ay: float) -> tuple[float, float]:
    """Convert normalized coordinates to iPhone Mirroring logical pixels."""
    x = ax / 1000 * WIN_W
    y = ay / 1000 * WIN_H
    return max(0, min(x, WIN_W - 1)), max(0, min(y, WIN_H_TAP_MAX))


def _find_iphone_window() -> tuple[float, float] | None:
    """Return (x, y) screen origin of the 318x701 iPhone Mirroring window."""
    windows = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
    for w in windows:
        owner = w.get("kCGWindowOwnerName", "")
        if "iPhone" not in owner:
            continue
        bounds = w.get("kCGWindowBounds", {})
        ww, wh = bounds.get("Width", 0), bounds.get("Height", 0)
        if int(ww) == 318 and int(wh) == 701:
            return (bounds["X"], bounds["Y"])
    return None


def _scroll_delta(direction: str) -> int:
    """Map scroll direction to CGEvent scroll wheel delta (pixels per tick).

    Positive = scroll up (content moves down, view earlier content).
    Negative = scroll down (content moves up, view later content).
    """
    if direction in ("up", "向上", "upward"):
        return SCROLL_DELTA
    if direction in ("down", "向下", "downward"):
        return -SCROLL_DELTA
    if direction in ("left", "向左", "leftward"):
        return -SCROLL_DELTA
    if direction in ("right", "向右", "rightward"):
        return SCROLL_DELTA
    raise ValueError(f"scroll direction must be up/down/left/right, got: {direction!r}")

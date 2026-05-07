"""Fixed action execution layer for policy experiments."""

import io
import time

from agent.utils import paste_text
from PIL import Image, ImageChops, ImageStat

from policy_expr.perception import LivePhoneSession
from policy_expr.schemas import Action, ActionDecision

WIN_W = 318
WIN_H = 701
WIN_H_TAP_MAX = WIN_H - 20   # 避开底部 Home 指示条安全区（约 20px）
SCROLL_DIFF_THRESHOLD = 8.0


class ActionExecutor:
    """Execute normalized policy actions via mirroir-mcp."""

    def __init__(self, phone: LivePhoneSession):
        self.phone = phone

    def execute(self, decision: ActionDecision, app_name: str = "") -> bool:
        action = decision.action
        print(f"\n动作: [{action.action_type}] {action.description}")

        if action.action_type == "tap" and action.x is not None and action.y is not None:
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
        gestures = scroll_gestures(direction)
        previous = self.phone.screenshot()

        for kind, args in gestures:
            from_x, from_y, to_x, to_y, duration_ms = args
            print(
                f"执行滚动({direction or action.direction})/{kind}: "
                f"({from_x:.0f},{from_y:.0f}) -> ({to_x:.0f},{to_y:.0f})"
            )
            if kind == "swipe":
                result = self._client().swipe(from_x, from_y, to_x, to_y, duration_ms=duration_ms)
            else:
                result = self._client().drag(from_x, from_y, to_x, to_y, duration_ms=duration_ms)
            time.sleep(0.8)
            current = self.phone.screenshot()
            mean_diff = mean_image_diff(previous, current)
            print(f"结果: {result}；mean_diff={mean_diff:.2f}")
            if mean_diff >= SCROLL_DIFF_THRESHOLD:
                print("滚动成功")
                return
            previous = current

        print("滚动可能未生效，页面可能已在边界")

    def _client(self):
        if not self.phone.client:
            raise RuntimeError("MCP 尚未连接")
        return self.phone.client


def logical_xy(ax: float, ay: float) -> tuple[float, float]:
    """Convert normalized coordinates to iPhone Mirroring logical pixels."""

    x = ax / 1000 * WIN_W
    y = ay / 1000 * WIN_H
    return clamp(x, 0, WIN_W - 1), clamp(y, 0, WIN_H_TAP_MAX)


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def mean_image_diff(before_png: bytes, after_png: bytes) -> float:
    before = Image.open(io.BytesIO(before_png)).convert("RGB")
    after = Image.open(io.BytesIO(after_png)).convert("RGB")
    diff = ImageChops.difference(before, after)
    return sum(ImageStat.Stat(diff).mean) / 3


def scroll_gestures(direction: str) -> list[tuple[str, tuple[float, float, float, float, int]]]:
    center_x = WIN_W * 0.50
    right_safe_x = WIN_W * 0.88
    upper_y = WIN_H * 0.33
    lower_y = WIN_H * 0.87
    center_y = WIN_H * 0.50
    left_x = WIN_W * 0.12
    right_x = WIN_W * 0.88

    if direction in ("up", "向上", "upward"):
        return [
            ("swipe", (center_x, lower_y, center_x, upper_y, 800)),
            ("swipe", (right_safe_x, lower_y, right_safe_x, upper_y, 1000)),
            ("drag", (center_x, lower_y, center_x, upper_y, 1600)),
            ("drag", (right_safe_x, lower_y, right_safe_x, upper_y, 1800)),
        ]
    if direction in ("down", "向下", "downward"):
        return [
            ("swipe", (center_x, upper_y, center_x, lower_y, 800)),
            ("swipe", (right_safe_x, upper_y, right_safe_x, lower_y, 1000)),
            ("drag", (center_x, upper_y, center_x, lower_y, 1600)),
            ("drag", (right_safe_x, upper_y, right_safe_x, lower_y, 1800)),
        ]
    if direction in ("left", "向左", "leftward"):
        return [
            ("swipe", (right_x, center_y, left_x, center_y, 600)),
            ("drag",  (right_x, center_y, left_x, center_y, 1200)),
        ]
    if direction in ("right", "向右", "rightward"):
        return [
            ("swipe", (left_x, center_y, right_x, center_y, 600)),
            ("drag",  (left_x, center_y, right_x, center_y, 1200)),
        ]
    raise ValueError(f"scroll direction must be up/down/left/right, got: {direction!r}")

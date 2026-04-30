"""Fixed action execution layer for policy experiments."""

import time

from agent.utils import paste_text
from policy_expr.perception import LivePhoneSession
from policy_expr.schemas import Action, PolicyDecision

WIN_W = 318
WIN_H = 701
SCROLL_SWIPE_FROM_Y_FRACTION = 0.56
SCROLL_SWIPE_TO_Y_FRACTION = 0.11


class ActionExecutor:
    """Execute normalized policy actions via mirroir-mcp."""

    def __init__(self, phone: LivePhoneSession):
        self.phone = phone

    def execute(self, decision: PolicyDecision) -> bool:
        action = decision.action
        print(f"\n动作: [{action.action_type}] {action.description}")

        if action.action_type == "tap" and action.x is not None and action.y is not None:
            lx, ly = logical_xy(action.x, action.y)
            self._tap(lx, ly, decision)

        elif (
            action.action_type == "type"
            and action.x is not None
            and action.y is not None
            and action.text
        ):
            lx, ly = logical_xy(action.x, action.y)
            self._tap(lx, ly, decision)
            time.sleep(0.5)
            print(f"输入文字: {action.text!r}")
            paste_text(action.text)
            print("结果: 已通过剪贴板粘贴输入")

        elif (
            action.action_type == "scroll"
            and action.x is not None
            and action.y is not None
            and action.direction
        ):
            self._scroll(action)

        elif action.action_type == "home":
            print("执行返回主屏")
            result = self._client().press_home()
            if "Failed to press Home" in result:
                print(f"结果: {result}，改为点击底部 Home 指示条")
                result = self._client().tap(WIN_W / 2, WIN_H - 16)
            print(f"结果: {result}")

        else:
            print(f"跳过执行：action_type={action.action_type!r}，需手动处理")
            return False

        return True

    def _tap(self, lx: float, ly: float, decision: PolicyDecision) -> None:
        action = decision.action
        in_wechat = (decision.app_name or "").strip() in ("微信", "WeChat")
        in_bottom_right = bool(action.x and action.y and action.x > 700 and action.y > 800)
        if in_wechat and in_bottom_right:
            print("检测到微信右下角点击，等待浮层消失...")
            time.sleep(2.0)
        print(f"执行点击: ({lx:.0f}, {ly:.0f})")
        result = self._client().tap(lx, ly)
        print(f"结果: {result}")

    def _scroll(self, action: Action) -> None:
        direction = (action.direction or "").strip().lower()
        cx = WIN_W / 2

        # Match mirroir-mcp's calibration/explorer scroll coordinates. Its
        # swipe tool posts scroll-wheel events at the midpoint, so the midpoint
        # must stay in the upper content area rather than near the tab/home bar.
        if direction in ("up", "向上", "upward"):
            fy = WIN_H * SCROLL_SWIPE_FROM_Y_FRACTION
            ty = WIN_H * SCROLL_SWIPE_TO_Y_FRACTION
        else:
            fy = WIN_H * SCROLL_SWIPE_TO_Y_FRACTION
            ty = WIN_H * SCROLL_SWIPE_FROM_Y_FRACTION

        print(f"执行滚动({direction or action.direction}): ({cx:.0f},{fy:.0f}) -> ({cx:.0f},{ty:.0f})")
        result = self._client().swipe(cx, fy, cx, ty, duration_ms=300)
        print(f"结果: {result}")

    def _client(self):
        if not self.phone.client:
            raise RuntimeError("MCP 尚未连接")
        return self.phone.client


def logical_xy(ax: float, ay: float) -> tuple[float, float]:
    """Convert normalized coordinates to iPhone Mirroring logical pixels."""

    x = ax / 1000 * WIN_W
    y = ay / 1000 * WIN_H
    return clamp(x, 0, WIN_W - 1), clamp(y, 0, WIN_H - 1)


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)

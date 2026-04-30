"""Fixed action execution layer for policy experiments."""

import time

from policy_expr.perception import LivePhoneSession
from policy_expr.schemas import Action, PolicyDecision

WIN_W = 318
WIN_H = 701


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
            result = self._client().type_text(action.text)
            print(f"结果: {result}")

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
        assert action.x is not None
        assert action.y is not None

        _, ly = logical_xy(action.x, action.y)
        offset = min(WIN_H * 0.35, 200)
        cx = WIN_W / 2
        if action.direction == "up":
            fy = min(ly + offset, WIN_H - 10)
            ty = max(ly - offset, 10)
        else:
            fy = max(ly - offset, 10)
            ty = min(ly + offset, WIN_H - 10)

        print(f"执行滚动({action.direction}): ({cx:.0f},{fy:.0f}) -> ({cx:.0f},{ty:.0f})")
        result = self._client().swipe(cx, fy, cx, ty, duration_ms=400)
        print(f"结果: {result}")

    def _client(self):
        if not self.phone.client:
            raise RuntimeError("MCP 尚未连接")
        return self.phone.client


def logical_xy(ax: float, ay: float) -> tuple[float, float]:
    """Convert normalized coordinates to iPhone Mirroring logical pixels."""

    return ax / 1000 * WIN_W, ay / 1000 * WIN_H

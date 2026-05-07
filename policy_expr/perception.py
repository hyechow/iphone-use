"""Environment sensing for policy experiments."""

import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from agent.sync_mcp_client import SyncMCPClient
from policy_expr.schemas import Observation

ROOT = Path(__file__).parent.parent
SCREENSHOT = ROOT / "logs" / "policy_expr" / "single-step" / "screenshot.png"

# macOS 截图 resize 到与 mcp 一致的尺寸（WIN_W*2 x WIN_H*2，2x Retina）
_TARGET_W = 636  # 318 * 2
_TARGET_H = 1402  # 701 * 2

# 截图后端：macos（不触发录屏检测）或 mcp
SCREENSHOT_BACKEND = "macos"


def _find_iphone_mirror_window() -> int | None:
    """Find iPhone Mirroring window ID via Quartz."""
    try:
        import Quartz
        for w in Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID
        ):
            if w.get("kCGWindowOwnerName") == "iPhone镜像":
                b = w.get("kCGWindowBounds", {})
                if b.get("Width") == 318 and b.get("Height") == 701:
                    return w.get("kCGWindowNumber")
    except Exception:
        pass
    return None


def _macos_screenshot() -> bytes:
    """Capture iPhone screen via macOS screencapture (bypasses screen-recording detection)."""
    wid = _find_iphone_mirror_window()
    if wid is None:
        raise RuntimeError("找不到 iPhone Mirroring 窗口（318×701），请确认已打开 iPhone镜像")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        subprocess.run(["screencapture", "-l", str(wid), path], check=True)
        full = Image.open(path).convert("RGB")
    finally:
        Path(path).unlink(missing_ok=True)

    # 裁掉设备外壳，取屏幕内容区域（扫描非黑像素边界）
    arr = np.array(full)
    mid_x, mid_y = arr.shape[1] // 2, arr.shape[0] // 2
    left  = next(x for x in range(arr.shape[1])          if arr[mid_y, x].max() > 20)
    right = next(x for x in range(arr.shape[1]-1, 0, -1) if arr[mid_y, x].max() > 20)
    top   = next(y for y in range(arr.shape[0])          if arr[y, mid_x].max() > 20)
    bot   = next(y for y in range(arr.shape[0]-1, 0, -1) if arr[y, mid_x].max() > 20)
    screen = full.crop((left, top, right + 1, bot + 1))

    # 贴到黑底画布，补回 mcp 自带的黑边，与 mcp 坐标系对齐
    # mcp 实测：左16、右17、上76、下17（Retina px）
    _PAD_L, _PAD_T = 16, 76
    _CONTENT_W = _TARGET_W - _PAD_L - 17   # 636 - 16 - 17 = 603
    _CONTENT_H = _TARGET_H - _PAD_T - 17   # 1402 - 76 - 17 = 1309
    screen = screen.resize((_CONTENT_W, _CONTENT_H), Image.LANCZOS)
    canvas = Image.new("RGB", (_TARGET_W, _TARGET_H), (0, 0, 0))
    canvas.paste(screen, (_PAD_L, _PAD_T))
    content = canvas

    buf = io.BytesIO()
    content.save(buf, format="PNG")
    return buf.getvalue()


class LivePhoneSession:
    """Own the mirroir-mcp connection used for sensing and execution."""

    def __init__(self):
        self.client: SyncMCPClient | None = None

    def __enter__(self) -> "LivePhoneSession":
        print("连接手机中...")
        self.client = SyncMCPClient()
        self.client.connect()
        print("MCP 连接成功")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.client:
            self.client.close()
        self.client = None

    def screenshot(self) -> bytes:
        if SCREENSHOT_BACKEND == "macos":
            return _macos_screenshot()
        if not self.client:
            raise RuntimeError("MCP 尚未连接")
        return self.client.screenshot()


class LivePerception:
    """Capture the current phone screen through an active live session."""

    def __init__(self, phone: LivePhoneSession, screenshot_path: Path = SCREENSHOT):
        self.phone = phone
        self.screenshot_path = screenshot_path

    def observe(self) -> Observation:
        print("截图中...")
        png_bytes = self.phone.screenshot()
        self.screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.screenshot_path.write_bytes(png_bytes)
        print(f"截图大小: {len(png_bytes) // 1024} KB，已保存到 {self.screenshot_path}")
        return Observation(png_bytes=png_bytes, source="live")

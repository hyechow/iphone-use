"""Environment sensing for policy experiments."""

import io
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from policy_expr.sync_mcp_client import SyncMCPClient
from policy_expr.schemas import Observation

ROOT = Path(__file__).parent.parent
SCREENSHOT = ROOT / "logs" / "policy_expr" / "single-step" / "screenshot.png"
_SCK_SERVER = ROOT / "bin" / "sck_server"
_MASK_PATH = Path(__file__).parent / "assets" / "mcp_frame_mask.png"

WIN_W = 318
WIN_H = 701

# Load mask once: 255 = device frame (black out), 0 = screen content (keep)
_FRAME_MASK: np.ndarray | None = (
    np.array(Image.open(_MASK_PATH).convert("L"))
    if _MASK_PATH.exists() else None
)


def _apply_mcp_frame(png_bytes: bytes) -> bytes:
    """Apply the MCP device-frame mask to a raw SCK screenshot.

    Pixels where the mask is non-zero are blacked out, matching mirror-mcp's
    device chrome (Dynamic Island, rounded corners, border) exactly.
    Falls back to returning the image unchanged if no mask file exists.
    """
    if _FRAME_MASK is None:
        return png_bytes
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(img)
    arr[_FRAME_MASK > 0] = 0
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class SCKSession:
    """Manages the SCK stream server subprocess for low-flash screenshot capture.

    The stream starts once (triggering the iOS recording indicator once).
    Subsequent screenshot() calls grab frames from the live stream silently.
    """

    def __init__(self):
        self._proc: subprocess.Popen | None = None

    def start(self):
        if not _SCK_SERVER.exists():
            raise FileNotFoundError(
                f"sck_server binary not found at {_SCK_SERVER}. "
                "Run: swiftc sck/sck_stream_server.swift ... -o bin/sck_server"
            )
        self._proc = subprocess.Popen(
            [str(_SCK_SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )
        # Wait for "ready\n" on stdout — confirms stream is up
        assert self._proc.stdout
        line = self._proc.stdout.readline()
        if line.strip() != b"ready":
            raise RuntimeError(f"SCK server failed to start: {line!r}")

    def screenshot(self, retries: int = 10) -> bytes:
        assert self._proc and self._proc.stdin and self._proc.stdout
        for _ in range(retries):
            self._proc.stdin.write(b"screenshot\n")
            self._proc.stdin.flush()
            raw = self._proc.stdout.read(4)
            if len(raw) < 4:
                raise RuntimeError("SCK server closed unexpectedly")
            (length,) = struct.unpack(">I", raw)
            if length > 0:
                return _apply_mcp_frame(self._proc.stdout.read(length))
            time.sleep(0.2)
        raise RuntimeError("SCK server: no frame available after retries")

    def close(self):
        if self._proc:
            self._proc.terminate()
            self._proc = None


class LivePhoneSession:
    """Own the mirroir-mcp connection used for execution, and SCK for screenshots."""

    def __init__(self):
        self.client: SyncMCPClient | None = None
        self._sck: SCKSession | None = None

    def __enter__(self) -> "LivePhoneSession":
        print("启动 SCK 截图流...")
        sck = SCKSession()
        sck.start()
        self._sck = sck
        print("SCK 截图流就绪")

        print("连接手机中...")
        self.client = SyncMCPClient()
        self.client.connect()
        print("MCP 连接成功")
        return self

    def __exit__(self, *_):
        if self._sck:
            self._sck.close()
            self._sck = None
        if self.client:
            self.client.close()
        self.client = None

    def screenshot(self) -> bytes:
        if not self._sck:
            raise RuntimeError("SCK 尚未连接")
        return self._sck.screenshot()


def mirroring_window_bounds() -> tuple[float, float, float, float] | None:
    """Return (x, y, w, h) of the iPhone mirroring window, or None if not found."""
    try:
        import Quartz
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )
        for w in windows:
            if "iphone" not in w.get("kCGWindowOwnerName", "").lower():
                continue
            b = w.get("kCGWindowBounds", {})
            if b.get("Width", 0) > 100 and b.get("Height", 0) > 400:
                return b["X"], b["Y"], b["Width"], b["Height"]
    except Exception:
        pass
    return None


def try_resume_mac(button_lx: float = WIN_W / 2, button_ly: float = WIN_H * 0.58) -> bool:
    """Click a dismiss button inside the Mac iPhone-mirroring window.

    Dismisses Mac-side system dialogs (e.g. "无法从 Mac 使用 iPhone 麦克风") that
    block iPhone interaction, causing client.tap() to return a 'paused' message.
    button_lx / button_ly: pixel offset from window top-left (defaults to ~center-x, 58%-down).
    """
    try:
        import Quartz
        bounds = mirroring_window_bounds()
        if bounds is None:
            return False
        win_x, win_y, _, _ = bounds
        pt = Quartz.CGPoint(win_x + button_lx, win_y + button_ly)
        for ev_type in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
            ev = Quartz.CGEventCreateMouseEvent(None, ev_type, pt, Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(1.0)
        return True
    except Exception:
        return False


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

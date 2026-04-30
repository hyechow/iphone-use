"""Environment sensing for policy experiments."""

from pathlib import Path

from agent.sync_mcp_client import SyncMCPClient
from policy_expr.schemas import Observation

ROOT = Path(__file__).parent.parent
SCREENSHOT = ROOT / "images" / "screenshot.png"


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
        self.screenshot_path.write_bytes(png_bytes)
        print(f"截图大小: {len(png_bytes) // 1024} KB，已保存到 {self.screenshot_path}")
        return Observation(png_bytes=png_bytes, source="live")

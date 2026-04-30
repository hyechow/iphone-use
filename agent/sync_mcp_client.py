"""Synchronous MCP client using subprocess + newline-delimited JSON-RPC."""

import base64
import json
import subprocess
import time


class SyncMCPClient:
    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._request_id = 0

    def connect(self):
        self._proc = subprocess.Popen(
            ["npx", "-y", "mirroir-mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        # Initialize handshake
        self._send({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "sync-client", "version": "0.1.0"},
            },
        })
        self._recv()  # consume init response
        # Initialized notification (no id, no response expected)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, msg: dict):
        data = json.dumps(msg) + "\n"
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(data.encode())
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        assert self._proc and self._proc.stdout
        line = self._proc.stdout.readline()
        if not line:
            raise ConnectionError("MCP server closed connection")
        return json.loads(line)

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """Call a mirroir-mcp tool and return the raw JSON-RPC result."""

        self._request_id += 1
        self._send({
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        })
        return self._recv()

    def screenshot(self) -> bytes:
        last_result = None
        for attempt in range(3):
            result = self.call_tool("screenshot")
            last_result = result
            for item in result.get("result", {}).get("content", []):
                if item.get("type") == "image" and item.get("data"):
                    return base64.b64decode(item["data"])
            if attempt < 2:
                time.sleep(0.4)
        raise RuntimeError(f"No image in screenshot response: {last_result}")

    def tap(self, x: float, y: float) -> str:
        result = self.call_tool("tap", {"x": x, "y": y})
        return text_content(result)

    def type_text(self, text: str) -> str:
        result = self.call_tool("type_text", {"text": text})
        return text_content(result)

    def swipe(
        self,
        from_x: float,
        from_y: float,
        to_x: float,
        to_y: float,
        duration_ms: int = 400,
    ) -> str:
        result = self.call_tool(
            "swipe",
            {
                "from_x": from_x,
                "from_y": from_y,
                "to_x": to_x,
                "to_y": to_y,
                "duration_ms": duration_ms,
            },
        )
        return text_content(result)

    def press_home(self) -> str:
        result = self.call_tool("press_home")
        return text_content(result)

    def close(self):
        if self._proc:
            self._proc.terminate()
            self._proc = None


def text_content(result: dict) -> str:
    """Extract plain text content from an MCP tools/call response."""

    if "error" in result:
        return str(result["error"])

    parts = []
    for item in result.get("result", {}).get("content", []):
        if item.get("type") == "text":
            parts.append(item["text"])
    return "\n".join(parts)

"""Synchronous MCP client using subprocess + newline-delimited JSON-RPC."""

import base64
import json
import subprocess


class SyncMCPClient:
    def __init__(self):
        self._proc: subprocess.Popen | None = None

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

    def screenshot(self) -> bytes:
        self._send({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "screenshot", "arguments": {}},
        })
        result = self._recv()
        for item in result.get("result", {}).get("content", []):
            if item.get("type") == "image" and item.get("data"):
                return base64.b64decode(item["data"])
        raise RuntimeError("No image in screenshot response")

    def tap(self, x: float, y: float) -> str:
        self._send({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "tap", "arguments": {"x": x, "y": y}},
        })
        result = self._recv()
        for item in result.get("result", {}).get("content", []):
            if item.get("type") == "text":
                return item["text"]
        return ""

    def close(self):
        if self._proc:
            self._proc.terminate()
            self._proc = None

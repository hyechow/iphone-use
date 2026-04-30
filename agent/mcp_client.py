import base64
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self):
        self._session: ClientSession | None = None
        self._exit_stack = None

    async def connect(self):
        from contextlib import AsyncExitStack

        server = StdioServerParameters(command="npx", args=["-y", "mirroir-mcp"])
        self._exit_stack = AsyncExitStack()
        r, w = await self._exit_stack.enter_async_context(stdio_client(server))
        self._session = await self._exit_stack.enter_async_context(ClientSession(r, w))
        await self._session.initialize()

    async def screenshot(self) -> bytes:
        assert self._session, "Not connected"
        last_content = None
        for attempt in range(3):
            result = await self._session.call_tool("screenshot", {})
            last_content = result.content
            for item in result.content:
                if item.type == "image" and hasattr(item, "data"):
                    return base64.b64decode(item.data)
            if attempt < 2:
                await asyncio.sleep(0.4)
        raise RuntimeError(f"No image in screenshot response: {last_content}")

    async def tap(self, x: float, y: float) -> str:
        assert self._session, "Not connected"
        result = await self._session.call_tool("tap", {"x": x, "y": y})
        return result.content[0].text if result.content else ""

    async def list_tools(self) -> list[str]:
        assert self._session, "Not connected"
        result = await self._session.list_tools()
        return [t.name for t in result.tools]

    async def close(self):
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

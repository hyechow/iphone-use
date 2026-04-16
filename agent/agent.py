import asyncio
import base64
from dataclasses import dataclass

from agent.mcp_client import MCPClient


@dataclass
class AgentEvent:
    type: str  # "screenshot" | "thinking" | "action" | "done" | "error"
    data: str  # text, or base64-encoded PNG for "screenshot"


class PhoneAgent:
    async def run(self, instruction: str, queue: asyncio.Queue[AgentEvent]):
        client = MCPClient()
        try:
            await queue.put(AgentEvent(type="thinking", data="正在连接手机..."))
            await client.connect()

            await queue.put(AgentEvent(type="thinking", data="截图中..."))
            png_bytes = await client.screenshot()
            b64 = base64.b64encode(png_bytes).decode()
            await queue.put(AgentEvent(type="screenshot", data=b64))

            # TODO: 接入真实 Claude agent 循环
            await queue.put(AgentEvent(type="thinking", data=f"收到指令：{instruction}"))
            await asyncio.sleep(0.5)
            await queue.put(AgentEvent(type="done", data="框架已就绪，agent 逻辑待实现"))

        except Exception as e:
            await queue.put(AgentEvent(type="error", data=str(e)))
        finally:
            await client.close()

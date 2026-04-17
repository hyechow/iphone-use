"""测试 LangGraph Agent 端到端流程：截图 → 定位 → 点击。

用法:
  uv run python scripts/agent_test.py "打开微信"
  uv run python scripts/agent_test.py "返回主屏幕"
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from agent.events import AgentEvent
from agent.runner import PhoneAgent

load_dotenv()

HEARTBEAT = 5   # 超过 N 秒没有事件时打印心跳
TIMEOUT   = 120  # 总超时


def ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


async def main(instruction: str):
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
    agent = PhoneAgent()
    thread_id = "test-001"

    print(f"{ts()} 指令: {instruction!r}\n", flush=True)

    task = asyncio.create_task(agent.run(thread_id, instruction, queue))
    deadline = time.monotonic() + TIMEOUT

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"\n{ts()} [超时] {TIMEOUT}s 内未完成")
            task.cancel()
            break

        try:
            event: AgentEvent = await asyncio.wait_for(
                queue.get(), timeout=min(HEARTBEAT, remaining)
            )
        except asyncio.TimeoutError:
            print(f"{ts()} [等待中...] queue 无事件", flush=True)
            continue

        if event.type == "screenshot":
            print(f"\n{ts()} [截图] 已获取（{len(event.data)} 字节 base64）", flush=True)
        elif event.type == "thinking":
            print(event.data, end="", flush=True)
        elif event.type == "action":
            print(f"\n{ts()} [动作] {event.data}", flush=True)
        elif event.type == "done":
            print(f"\n{ts()} [完成] {event.data}", flush=True)
            break
        elif event.type == "error":
            print(f"\n{ts()} [错误] {event.data}", flush=True)
            break

    await task


if __name__ == "__main__":
    instruction = sys.argv[1] if len(sys.argv) > 1 else "打开微信"
    asyncio.run(main(instruction))

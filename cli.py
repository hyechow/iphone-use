"""Terminal REPL for iPhone agent interaction."""

import asyncio

from rich.console import Console

from agent.events import AgentEvent
from agent.runner import PhoneAgent

console = Console()

EVENT_ICONS = {
    "thinking":   "💭",
    "screenshot": "📸",
    "action":     "👆",
    "done":       "✅",
    "error":      "❌",
}


async def consume_events(queue: asyncio.Queue[AgentEvent | None]):
    """Print agent events as they arrive. Returns on None sentinel."""
    buffer = ""
    while True:
        event: AgentEvent | None = await queue.get()
        if event is None:
            return

        if event.type == "thinking":
            buffer += event.data
            if "\n" in buffer:
                lines = buffer.split("\n")
                for line in lines[:-1]:
                    console.print(f"  {EVENT_ICONS.get('thinking', '')} {line}", style="dim italic")
                buffer = lines[-1]
        else:
            if buffer.strip():
                console.print(f"  {EVENT_ICONS.get('thinking', '')} {buffer}", style="dim italic")
                buffer = ""

            icon = EVENT_ICONS.get(event.type, "")
            if event.type == "screenshot":
                console.print(f"  {icon} 截图已更新", style="green")
            elif event.type == "done":
                console.print(f"\n  {icon} {event.data}\n", style="bold green")
                return
            elif event.type == "error":
                console.print(f"\n  {icon} {event.data}\n", style="bold red")
                return
            else:
                console.print(f"  {icon} {event.data}", style="yellow")


async def run_agent(thread_id: str, instruction: str):
    """Run agent and display events."""
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
    agent = PhoneAgent()

    task = asyncio.create_task(agent.run(thread_id, instruction, queue))
    try:
        await consume_events(queue)
        await task
    except asyncio.CancelledError:
        console.print("\n  ⏹ 已中断", style="yellow")


async def repl():
    import uuid

    thread_id: str | None = None
    console.print("[bold]iPhone Agent CLI[/bold]  输入指令控制手机，/new 新对话，/quit 退出\n")

    while True:
        try:
            instruction = await asyncio.to_thread(input, "❯ ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见 👋")
            break

        instruction = instruction.strip()
        if not instruction:
            continue
        if instruction == "/quit":
            console.print("再见 👋")
            break
        if instruction == "/new":
            thread_id = None
            console.print("[yellow]已开始新对话[/yellow]\n")
            continue

        if thread_id is None:
            thread_id = str(uuid.uuid4())

        console.print()
        await run_agent(thread_id, instruction)


def main():
    asyncio.run(repl())


if __name__ == "__main__":
    main()

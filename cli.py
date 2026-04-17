"""Terminal REPL for iPhone agent interaction."""

import os

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agent.events import AgentEvent
from agent.runner import PhoneAgent

console = Console()
_width = os.get_terminal_size().columns

EVENT_ICONS = {
    "thinking":   "💭",
    "reasoning":  "🧠",
    "screenshot": "📸",
    "action":     "👆",
    "done":       "✅",
    "error":      "❌",
}


def _separator():
    console.print(f"[dim]{'─' * _width}[/dim]")


class EventDisplay:
    """Stateful display for streaming agent events."""

    def __init__(self):
        self.buffer = ""
        self.thinking_lines: list[str] = []

    def _flush_thinking(self):
        if self.buffer.strip():
            self.thinking_lines.append(self.buffer)
            self.buffer = ""
        if self.thinking_lines:
            console.print()
            for line in self.thinking_lines:
                console.print(f"  [dim italic]{line}[/dim italic]")
            self.thinking_lines.clear()

    def process(self, event: AgentEvent):
        if event.type == "reasoning":
            self._flush_thinking()
            console.print()
            console.print(f"  [dim]{EVENT_ICONS['reasoning']} 推理过程:[/dim]")
            for line in event.data.strip().split("\n"):
                if line.strip():
                    console.print(f"    [dim]{line}[/dim]")
        elif event.type == "thinking":
            self.buffer += event.data
            if "\n" in self.buffer:
                lines = self.buffer.split("\n")
                self.thinking_lines.extend(lines[:-1])
                self.buffer = lines[-1]
        else:
            self._flush_thinking()
            icon = EVENT_ICONS.get(event.type, "")
            if event.type == "screenshot":
                console.print()
                console.print(f"  {icon} [green]截图已更新[/green]")
            elif event.type == "done":
                console.print()
                console.print(f"  {icon} [bold green]{event.data}[/bold green]")
                _separator()
            elif event.type == "error":
                console.print()
                console.print(f"  {icon} [bold red]{event.data}[/bold red]")
                _separator()
            else:
                console.print(f"  {icon} [yellow]{event.data}[/yellow]")

    def flush(self):
        self._flush_thinking()


def main():
    import uuid

    agent = PhoneAgent()
    display = EventDisplay()
    thread_id: str | None = None

    console.print(Panel(
        "[bold]iPhone Agent CLI[/bold]\n"
        "输入指令控制手机 | [cyan]/new[/cyan] 新对话 | [cyan]/quit[/cyan] 退出",
        border_style="bright_black",
        padding=(0, 1),
    ))

    while True:
        try:
            instruction = input("\n❯ ")
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
            console.print("[yellow]已开始新对话[/yellow]")
            _separator()
            continue

        if thread_id is None:
            thread_id = str(uuid.uuid4())

        for event in agent.run(thread_id, instruction):
            display.process(event)


if __name__ == "__main__":
    main()

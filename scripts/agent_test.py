"""iPhone Agent 交互式终端。

用法:
  uv run python scripts/agent_test.py          # 交互模式
  uv run python scripts/agent_test.py "打开微信"  # 单次执行
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from agent.events import AgentEvent
from agent.runner import PhoneAgent

load_dotenv()


def ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


def run_instruction(instruction: str, thread_id: str = "cli-001"):
    agent = PhoneAgent()
    print(f"\n{ts()} 指令: {instruction!r}\n", flush=True)

    try:
        for event in agent.run(thread_id, instruction):
            if event.type == "screenshot":
                print(f"\n{ts()} [截图] 已获取（{len(event.data)} 字节 base64）", flush=True)
            elif event.type == "reasoning":
                print(f"\n\033[2m{event.data}\033[0m", end="", flush=True)
            elif event.type == "thinking":
                print(event.data, end="", flush=True)
            elif event.type == "action":
                print(f"\n{ts()} [动作] {event.data}", flush=True)
            elif event.type == "done":
                print(f"\n\n{ts()} [完成]", flush=True)
            elif event.type == "error":
                print(f"\n{ts()} [错误] {event.data}", flush=True)
    except KeyboardInterrupt:
        print(f"\n{ts()} [中断]", flush=True)


def repl():
    print("iPhone Agent  (Ctrl+D 退出，Ctrl+C 中断当前任务)\n")
    thread_id = "cli-001"
    history = InMemoryHistory()
    while True:
        try:
            instruction = prompt(">>> ", history=history).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if not instruction:
            continue
        run_instruction(instruction, thread_id)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_instruction(" ".join(sys.argv[1:]))
    else:
        repl()

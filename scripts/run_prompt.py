"""Run one prompt against the real iPhone agent.

Usage:
  uv run python scripts/run_prompt.py "打开微信"
  uv run python scripts/run_prompt.py "打开微信" --thread-id test-session
"""
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.runner import PhoneAgent


def main():
    parser = argparse.ArgumentParser(description="Run one prompt against the real iPhone agent")
    parser.add_argument("instruction", help="Natural language instruction for the agent")
    parser.add_argument("--thread-id", default=None, help="Optional LangGraph thread/session id")
    args = parser.parse_args()

    thread_id = args.thread_id or str(uuid.uuid4())
    agent = PhoneAgent()

    print(f"thread_id: {thread_id}")
    print(f"instruction: {args.instruction}")

    for event in agent.run(thread_id, args.instruction):
        if event.type == "screenshot":
            print(f"[screenshot] base64_png_chars={len(event.data)}")
        else:
            print(f"[{event.type}] {event.data}")


if __name__ == "__main__":
    main()

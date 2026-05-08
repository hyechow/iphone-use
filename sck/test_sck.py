"""Test SCK-based screenshot — indicator should only flash once at startup."""

import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent  # project root
sys.path.insert(0, str(ROOT))
from policy_expr.perception import _apply_mcp_frame
SCK_SERVER = ROOT / "bin" / "sck_server"
OUT = Path("/tmp/sck_test.png")


def start_server() -> subprocess.Popen:
    return subprocess.Popen(
        [str(SCK_SERVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
    )


def screenshot(proc: subprocess.Popen, retries: int = 15) -> bytes:
    assert proc.stdin and proc.stdout
    for _ in range(retries):
        proc.stdin.write(b"screenshot\n")
        proc.stdin.flush()
        raw = proc.stdout.read(4)
        if len(raw) < 4:
            raise RuntimeError("Server closed")
        (length,) = struct.unpack(">I", raw)
        if length > 0:
            return proc.stdout.read(length)
        time.sleep(0.2)
    raise RuntimeError("No frame after retries")


def main():
    print("Starting SCK server (indicator should appear ONCE)...")
    proc = start_server()

    print("Waiting for stream to start...")
    line = proc.stdout.readline()
    if line.strip() != b"ready":
        print(f"Unexpected: {line!r}")
        proc.terminate()
        return

    print("Stream ready. Taking 5 screenshots with 2s intervals...")
    print("Watch the iPhone — indicator should NOT flash after the first time.\n")

    for i in range(5):
        png = screenshot(proc)
        OUT.write_bytes(_apply_mcp_frame(png))
        print(f"  [{i+1}/5] {len(png)//1024} KB → {OUT}")
        if i < 4:
            time.sleep(2.0)

    proc.terminate()
    print("\nDone. Open /tmp/sck_test.png to check the last frame.")


if __name__ == "__main__":
    main()

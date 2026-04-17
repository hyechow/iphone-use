import base64
import io
import subprocess
import time

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from PIL import Image

from agent.sessions import get_client
from agent.utils import home_indicator_coords


@tool
def take_screenshot(config: RunnableConfig) -> str:
    """Take a screenshot of the current iPhone screen. Returns a base64-encoded PNG image."""
    session_id = config["configurable"]["thread_id"]
    client = get_client(session_id)
    png_bytes = client.screenshot()
    # Retina screenshots are 2×; resize to logical pixels to reduce token cost
    img = Image.open(io.BytesIO(png_bytes))
    small = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@tool
def tap_screen(x: float, y: float, config: RunnableConfig) -> str:
    """Tap a position on the iPhone screen.

    Args:
        x: Normalized x coordinate (0-1000, left=0, right=1000)
        y: Normalized y coordinate (0-1000, top=0, bottom=1000)
    """
    session_id = config["configurable"]["thread_id"]
    client = get_client(session_id)
    # Convert normalized 0-1000 → logical pixels (window is 318×701)
    lx = x / 1000 * 318
    ly = y / 1000 * 701
    result = client.tap(lx, ly)
    return result or f"Tapped at ({lx:.0f}, {ly:.0f})"


@tool
def go_to_home_screen(config: RunnableConfig) -> str:
    """Return to the iPhone home screen by tapping the home indicator bar at the bottom."""
    session_id = config["configurable"]["thread_id"]
    client = get_client(session_id)
    # Window size is 318x701 for the standard iPhone Mirroring window
    x, y = home_indicator_coords(318, 701)
    result = client.tap(x, y)
    return result or "Tapped home indicator"



@tool
def type_text(text: str) -> str:
    """Type text into the currently focused input field on the iPhone.

    Uses clipboard paste (pbcopy + Cmd+V) to support Chinese and all Unicode.

    Args:
        text: The text to type into the input field.
    """
    subprocess.run(["pbcopy"], input=text.encode(), check=True)
    time.sleep(0.1)
    subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to keystroke "v" using command down',
    ], check=True)
    return f"Typed: {text!r}"


TOOLS = [take_screenshot, tap_screen, go_to_home_screen, type_text]

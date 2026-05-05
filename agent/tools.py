import base64
import io
import subprocess
import time
from collections.abc import Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from PIL import Image, ImageChops, ImageStat

from agent.sessions import get_client
from agent.utils import home_indicator_coords


def _require_coordinate(name: str, value: float | int | None) -> float:
    if value is None:
        raise ValueError(f"{name} is required for this tool call")
    number = float(value)
    if not 0 <= number <= 1000:
        raise ValueError(f"{name} must be between 0 and 1000")
    return number


def _paste_text(text: str) -> None:
    subprocess.run(["pbcopy"], input=text.encode(), check=True)
    time.sleep(0.1)
    subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to keystroke "v" using command down',
    ], check=True)


def _press_enter() -> None:
    time.sleep(0.1)
    subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to key code 36',
    ], check=True)


def _mean_image_diff(before_png: bytes, after_png: bytes) -> float:
    before = Image.open(io.BytesIO(before_png)).convert("RGB")
    after = Image.open(io.BytesIO(after_png)).convert("RGB")
    diff = ImageChops.difference(before, after)
    return sum(ImageStat.Stat(diff).mean) / 3


def _scroll_gestures(
    direction: str,
) -> list[tuple[str, tuple[float, float, float, float, int]]]:
    w, h = 318, 701
    center_x = w * 0.50
    right_safe_x = w * 0.88
    left_safe_x = w * 0.20
    upper_y = h * 0.33
    lower_y = h * 0.87
    left_x = w * 0.12
    right_x = w * 0.88
    mid_y = h * 0.55

    if direction == "up":
        return [
            ("swipe", (center_x, lower_y, center_x, upper_y, 800)),
            ("swipe", (right_safe_x, lower_y, right_safe_x, upper_y, 1000)),
            ("drag", (center_x, lower_y, center_x, upper_y, 1600)),
            ("drag", (right_safe_x, lower_y, right_safe_x, upper_y, 1800)),
        ]
    if direction == "down":
        return [
            ("swipe", (center_x, upper_y, center_x, lower_y, 800)),
            ("swipe", (right_safe_x, upper_y, right_safe_x, lower_y, 1000)),
            ("drag", (center_x, upper_y, center_x, lower_y, 1600)),
            ("drag", (right_safe_x, upper_y, right_safe_x, lower_y, 1800)),
        ]
    if direction == "left":
        return [
            ("swipe", (right_x, mid_y, left_x, mid_y, 700)),
            ("drag", (right_x, mid_y, left_x, mid_y, 1400)),
        ]
    if direction == "right":
        return [
            ("swipe", (left_x, mid_y, right_x, mid_y, 700)),
            ("drag", (left_x, mid_y, right_x, mid_y, 1400)),
        ]
    raise ValueError("direction must be one of: up, down, left, right")


def _call_gesture(
    kind: str,
    args: tuple[float, float, float, float, int],
    swipe: Callable[..., str],
    drag: Callable[..., str],
) -> str:
    from_x, from_y, to_x, to_y, duration_ms = args
    if kind == "swipe":
        return swipe(from_x, from_y, to_x, to_y, duration_ms=duration_ms)
    return drag(from_x, from_y, to_x, to_y, duration_ms=duration_ms)


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
    """Tap a single center point on the iPhone screen.

    Args:
        x: A single normalized x coordinate number (0-1000, left=0, right=1000). Do not pass a list or range.
        y: A single normalized y coordinate number (0-1000, top=0, bottom=1000). Do not pass a list or range.
    """
    session_id = config["configurable"]["thread_id"]
    client = get_client(session_id)
    x = _require_coordinate("x", x)
    y = _require_coordinate("y", y)
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
def type_text(text: str, press_enter: bool = True) -> str:
    """Type text into the currently focused input field on the iPhone.

    Uses clipboard paste (pbcopy + Cmd+V) to support Chinese and all Unicode.

    Args:
        text: The text to type into the input field.
        press_enter: Press Return after typing. Use this when the input should be submitted
            immediately, such as search fields or screens without a visible send button.
    """
    _paste_text(text)
    if press_enter:
        _press_enter()
        return f"Typed and pressed Enter: {text!r}"
    return f"Typed: {text!r}"


@tool
def tap_and_type(x: float, y: float, text: str, config: RunnableConfig, press_enter: bool = True) -> str:
    """Tap an input field and immediately type text into it.

    Use this instead of calling tap_screen + type_text separately whenever
    the goal is to enter text into a field (search bar, chat box, etc.).

    Args:
        x: Normalized x coordinate of the input field center (0-1000).
        y: Normalized y coordinate of the input field center (0-1000).
        text: The text to type after tapping.
        press_enter: Press Return after typing. Use this when the input should be submitted
            immediately, such as search fields or screens without a visible send button.
    """
    session_id = config["configurable"]["thread_id"]
    client = get_client(session_id)
    x = _require_coordinate("x", x)
    y = _require_coordinate("y", y)
    lx = x / 1000 * 318
    ly = y / 1000 * 701
    client.tap(lx, ly)
    time.sleep(0.3)  # wait for keyboard to appear
    _paste_text(text)
    if press_enter:
        _press_enter()
        return f"Tapped ({lx:.0f}, {ly:.0f}), typed {text!r}, and pressed Enter"
    return f"Tapped ({lx:.0f}, {ly:.0f}) and typed: {text!r}"


@tool
def scroll_screen(
    config: RunnableConfig,
    direction: str = "up",
    max_attempts: int = 4,
) -> str:
    """Scroll the current iPhone page.

    Use this when the current page needs to move up, down, left, or right.

    Args:
        direction: One of up, down, left, right. "up" means finger swipes upward,
            usually revealing lower content in vertical lists.
        max_attempts: Maximum attempts before reporting that the page did not move.
    """
    if direction not in {"up", "down", "left", "right"}:
        raise ValueError("direction must be one of: up, down, left, right")

    session_id = config["configurable"]["thread_id"]
    client = get_client(session_id)
    previous = client.screenshot()

    gestures = _scroll_gestures(direction)[: max(1, max_attempts)]
    for kind, gesture_args in gestures:
        _call_gesture(kind, gesture_args, client.swipe, client.drag)
        time.sleep(0.8)
        current = client.screenshot()
        mean_diff = _mean_image_diff(previous, current)
        if mean_diff >= 8.0:
            return f"Scrolled {direction}."
        previous = current

    return f"Could not scroll {direction}; the page may already be at the edge."


TOOLS = [tap_screen, go_to_home_screen, type_text, tap_and_type, scroll_screen]

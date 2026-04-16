import base64

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agent.sessions import get_client


@tool
async def take_screenshot(config: RunnableConfig) -> str:
    """Take a screenshot of the current iPhone screen. Returns a base64-encoded PNG image."""
    session_id = config["configurable"]["thread_id"]
    client = await get_client(session_id)
    png_bytes = await client.screenshot()
    return base64.b64encode(png_bytes).decode()


TOOLS = [take_screenshot]

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import base64
import io

from agent.events import AgentEvent
from agent.mcp_client import MCPClient
from agent.runner import PhoneAgent
from PIL import Image

app = FastAPI()

# Shared MCP client for independent screenshot polling
_screenshot_client: MCPClient | None = None


async def get_screenshot_client() -> MCPClient:
    global _screenshot_client
    if _screenshot_client is None or _screenshot_client._session is None:
        _screenshot_client = MCPClient()
        await _screenshot_client.connect()
    return _screenshot_client

FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"

# task_id -> asyncio.Queue
_task_queues: dict[str, asyncio.Queue[AgentEvent | None]] = {}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FRONTEND.read_text()


@app.get("/api/screenshot")
async def api_screenshot():
    client = await get_screenshot_client()
    png_bytes = await client.screenshot()
    img = Image.open(io.BytesIO(png_bytes))
    small = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image": b64}


class RunRequest(BaseModel):
    instruction: str
    thread_id: str | None = None


@app.post("/api/run")
async def api_run(req: RunRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
    _task_queues[task_id] = queue

    async def _run():
        agent = PhoneAgent()
        await agent.run(thread_id, req.instruction, queue)
        await queue.put(None)  # sentinel: stream end

    asyncio.create_task(_run())
    return {"task_id": task_id, "thread_id": thread_id}


@app.get("/api/stream/{task_id}")
async def api_stream(task_id: str):
    queue = _task_queues.get(task_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        try:
            while True:
                event: AgentEvent | None = await queue.get()
                if event is None:
                    break
                yield {"data": json.dumps({"type": event.type, "data": event.data})}
        finally:
            _task_queues.pop(task_id, None)

    return EventSourceResponse(event_generator())

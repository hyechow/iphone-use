import asyncio
import base64
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agent.agent import AgentEvent, PhoneAgent
from agent.mcp_client import MCPClient

app = FastAPI()

FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"

# task_id -> asyncio.Queue
_task_queues: dict[str, asyncio.Queue[AgentEvent | None]] = {}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FRONTEND.read_text()


@app.get("/api/screenshot")
async def api_screenshot():
    client = MCPClient()
    try:
        await client.connect()
        png_bytes = await client.screenshot()
        return {"image": base64.b64encode(png_bytes).decode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await client.close()


class RunRequest(BaseModel):
    instruction: str


@app.post("/api/run")
async def api_run(req: RunRequest):
    task_id = str(uuid.uuid4())
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
    _task_queues[task_id] = queue

    async def _run():
        agent = PhoneAgent()
        await agent.run(req.instruction, queue)
        await queue.put(None)  # sentinel: stream end

    asyncio.create_task(_run())
    return {"task_id": task_id}


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

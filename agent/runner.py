import asyncio

from langchain_core.messages import HumanMessage

from agent.agent import AgentEvent, _graph


class PhoneAgent:
    async def run(self, session_id: str, instruction: str, queue: asyncio.Queue[AgentEvent]):
        config = {"configurable": {"thread_id": session_id}}
        input_state = {
            "messages": [HumanMessage(content=instruction)],
        }

        try:
            async for event in _graph.astream_events(input_state, config, version="v2"):
                kind = event["event"]
                name = event.get("name", "")

                # 工具调用完成：截图结果推给前端
                if kind == "on_tool_end" and name == "take_screenshot":
                    b64 = event["data"].get("output", "")
                    if b64:
                        await queue.put(AgentEvent(type="screenshot", data=b64))

                # LLM token 流
                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    token = chunk.content
                    if isinstance(token, str) and token:
                        await queue.put(AgentEvent(type="thinking", data=token))
                    elif isinstance(token, list):
                        for part in token:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text = part.get("text", "")
                                if text:
                                    await queue.put(AgentEvent(type="thinking", data=text))

            await queue.put(AgentEvent(type="done", data="完成"))

        except Exception as e:
            await queue.put(AgentEvent(type="error", data=str(e)))

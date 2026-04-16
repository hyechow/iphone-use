"""测试 qwen-plus 是否支持图像输入"""
import asyncio
import base64
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from llm.provider_config import resolve_chat_provider_config

load_dotenv()


async def test_vision():
    # 使用 dashscope provider
    cfg = resolve_chat_provider_config(provider="modelscope", model="Qwen/Qwen3.5-35B-A3B")
    print(f"Provider : {cfg.provider}")
    print(f"Model    : {cfg.model}")
    print(f"Base URL : {cfg.base_url}")

    llm = ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )

    # 用已有截图，没有就跳过图像
    screenshot_path = Path("screenshot.png")
    if screenshot_path.exists():
        b64 = base64.b64encode(screenshot_path.read_bytes()).decode()
        content = [
            {"type": "text", "text": "这是一张截图，请描述你看到的内容。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
        print("\n[带图像]\n")
    else:
        content = "你好，请用一句话介绍自己。"
        print("\n[纯文本，screenshot.png 不存在]\n")

    response = await llm.ainvoke([HumanMessage(content=content)])
    print("回复：", response.content)


asyncio.run(test_vision())

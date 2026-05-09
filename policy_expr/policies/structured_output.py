"""Structured-output multimodal LLM action policy."""

import base64

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from policy_expr.config import resolve_llm_config
from llm.structured import invoke_structured
from policy_expr.policies.base import BaseActionPolicy, resize_to_logical_png
from policy_expr.schemas import ActionDecision, Observation

load_dotenv()


SYSTEM_PROMPT = """\
你是一个 iPhone 操作执行器。
用户会提供 iPhone 截图和一个具体的操作指令。你只需要找到目标元素并输出对应动作。

坐标使用归一化坐标系：左上角(0,0)，右下角(1000,1000)。
需要在输入框中输入文字时，使用 type（而非 tap），并填写输入框中心的 x/y 坐标和 text，它会自动先点击再输入。
只有当操作指令明确说明输入框已经聚焦时，type 可以只填写 text、不填写 x/y。
需要滚动页面时，使用 scroll，填写 direction 和 x/y 坐标。
- down：向下滚动，查看页面下方的内容
- up：向上滚动，查看页面上方的内容
- left：向左滚动，查看右侧内容（翻到下一页）
- right：向右滚动，查看左侧内容（翻到上一页）
多数按时间倒序的列表页新内容在顶部，要查看更早/更旧的内容通常选择 down；若截图显示相反方向，应以当前 UI 结构为准。
x/y 是滚轮事件的鼠标落点坐标，决定了滚轮作用于哪个 UI 元素：
- 整页滚动：x/y 放在可滚动内容的中部
- 滚轮选择器（日期、时间、城市等多列选择 UI）：x/y 必须精确落在目标列的区域上（如年份列偏左、月份列偏右），y 放在该列选中行的中间位置
y 坐标范围严格限制在 200-850 之间，禁止使用 y<200 或 y>850
需要返回主屏幕时，使用 home，无需填写坐标。
⚠️ home 只用于「明确需要退出当前应用回到桌面」的场景。如果目标元素在当前页面不可见，应优先寻找应用内的导航路径（如左上角返回按钮、底部 tab），而不是直接 home。
如果指令含义是「停止操作」「无需操作」「目标已完成」，使用 stop，无需填写任何坐标或文字。
action 的 description 用中文简要说明操作目标即可。
"""


class StructuredOutputPolicy(BaseActionPolicy):
    """Vision-based action policy: LLM screenshot analysis + structured output."""

    name = "structured_output"

    def decide(self, observation: Observation, instruction: str) -> ActionDecision:
        cfg = resolve_llm_config("action_policy")
        print(f"Provider : {cfg.provider}")
        print(f"Model    : {cfg.model}")

        b64 = base64.b64encode(resize_to_logical_png(observation.png_bytes)).decode()
        llm = ChatOpenAI(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"操作指令：{instruction}\n\n请根据截图执行该指令。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ]
            ),
        ]
        return invoke_structured(llm, messages, ActionDecision)

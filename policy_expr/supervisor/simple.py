"""Simple LLM-based supervisor policy."""

import base64

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from llm.structured import invoke_structured
from policy_expr.config import resolve_llm_config
from policy_expr.policies.base import resize_to_logical_png
from policy_expr.schemas import Observation, PolicyTurn, SupervisorStep

load_dotenv()


SYSTEM_PROMPT = """\
你是 iPhone 自动化任务的监督者（supervisor）。
每一轮你会收到：当前屏幕截图、用户目标、已执行的历史记录。

你的职责：
1. 审查历史，评估上一步动作是否成功执行并产生预期效果
2. 判断用户目标是否已完全达成（goal_completed=true/false）
3. 判断是否应该继续执行动作（should_act=true/false）
   - 目标已达成 → should_act=false, stop=true
   - 出现无法恢复的错误状态 → should_act=false, stop=true
   - 卡住或连续失败超过 3 次 → should_act=false, stop=true
   - 否则 should_act=true
4. 如果 should_act=true，给出精确的单步操作指令（instruction）
   - 描述具体要操作的目标元素和位置，如「点击底部导航栏左起第二个通讯录图标」
   - 不要给出目标级指令如「进入通讯录页面」
   - 参考历史中的失败记录，避免重复同样的错误操作
5. summary 用一句话描述当前屏幕状态和任务进展

请用中文填写所有描述性字段。
"""


class SimpleSupervisorPolicy:
    """LLM-based supervisor: assesses state and steers the action policy."""

    name = "simple"

    def step(
        self,
        observation: Observation,
        goal: str,
        history: list[PolicyTurn],
    ) -> SupervisorStep:
        cfg = resolve_llm_config("supervisor")
        print(f"Supervisor Provider : {cfg.provider}")
        print(f"Supervisor Model    : {cfg.model}")

        llm = ChatOpenAI(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )

        b64 = base64.b64encode(resize_to_logical_png(observation.png_bytes)).decode()
        history_text = _format_history(history)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"用户目标：{goal}\n\n"
                            f"历史记录（共 {len(history)} 轮）：\n{history_text}\n\n"
                            "请根据当前屏幕截图和历史记录，做出本轮监督决策。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ]
            ),
        ]
        return invoke_structured(llm, messages, SupervisorStep)


def _format_history(history: list[PolicyTurn]) -> str:
    if not history:
        return "（无历史记录，这是第一轮）"

    lines = []
    for turn in history[-8:]:
        sv = turn.supervisor
        if turn.action_decision and turn.executed:
            action = turn.action_decision.action
            lines.append(
                f"{turn.index}. [执行] [{action.action_type}] {action.description}"
                f"（{sv.summary}）"
            )
        elif turn.action_decision and not turn.executed:
            action = turn.action_decision.action
            lines.append(
                f"{turn.index}. [未执行] [{action.action_type}] {action.description}"
            )
        else:
            lines.append(f"{turn.index}. [跳过动作] {sv.summary}")
    return "\n".join(lines)

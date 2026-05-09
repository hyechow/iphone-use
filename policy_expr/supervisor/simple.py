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

## 目标完成校验规则
判断 goal_completed 时必须严格验证：
- 如果目标包含特定条件（如时间范围"本周""今天"、金额范围、特定类别），截图中可见的数据必须满足这些条件才能标 goal_completed=true
- 例：目标是"本周花了多少"，但页面显示的是月度数据 → goal_completed=false，需继续操作切换到周维度或提取符合时间范围的数据
- 例：目标是"最近的订单"，但订单列表没有时间标记 → goal_completed=false，需继续查看确认时间
- 不要因为页面上"有数据"就判定完成，必须确认数据的范围和维度与目标匹配
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


# ── CLI entry point: uv run python -m policy_expr.supervisor.simple "goal" ──


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from policy_expr.perception import LivePhoneSession

    goal = sys.argv[1] if len(sys.argv) > 1 else "打开微信"
    print(f"Goal: {goal}\n")

    sup = SimpleSupervisorPolicy()

    with LivePhoneSession() as phone:
        png_bytes = phone.screenshot()
        observation = Observation(png_bytes=png_bytes, source="live")

        log_dir = Path(__file__).parent.parent.parent / "logs" / "policy_expr" / "test"
        log_dir.mkdir(parents=True, exist_ok=True)
        shot_path = log_dir / "screenshot.png"
        shot_path.write_bytes(png_bytes)
        print(f"截图已保存: {shot_path}\n")

        sv = sup.step(observation, goal, [])
        print(f"should_act     : {sv.should_act}")
        print(f"instruction    : {sv.instruction}")
        print(f"stop           : {sv.stop}")
        print(f"goal_completed : {sv.goal_completed}")
        print(f"app_name       : {sv.app_name}")
        print(f"summary        : {sv.summary}")

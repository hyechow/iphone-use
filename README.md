# iphone-use

通过 MCP 协议控制 iPhone 的 AI 助手，支持自然语言操作并实时展示操作过程。

## 环境准备

```bash
# 安装 mirroir-mcp（iPhone 控制 MCP server）
brew tap jfarcand/tap
npx -y mirroir-mcp install

# 安装 Python 依赖
uv sync
```

## 配置

复制 `.env` 并填写 API 配置：

```bash
# 选择 LLM 提供商（modelscope / dashscope / nvidia / openai / local）
API_PROVIDER=modelscope

# ModelScope
MODELSCOPE_API_KEY=your_api_key
MODELSCOPE_MODEL=Qwen/Qwen3.5-35B-A3B  # 支持视觉输入

# 其他提供商按需配置
# NVIDIA_API_KEY=...
# NVIDIA_MODEL=moonshotai/kimi-k2-instruct

# 可选：为不同节点单独配置 LLM；未设置时回退到上面的全局配置
# PLAN_API_PROVIDER=modelscope
# PLAN_MODEL=Qwen/Qwen3.5-35B-A3B
# EXECUTE_API_PROVIDER=modelscope
# EXECUTE_MODEL=Qwen/Qwen3.5-35B-A3B
# CHECK_API_PROVIDER=modelscope
# CHECK_MODEL=Qwen/Qwen3.5-35B-A3B
```

## 启动服务

```bash
uv run uvicorn backend.main:app --reload
```

浏览器打开 http://localhost:8000，即可看到三栏界面（手机截图 / 对话 / 步骤流）。

## 测试

```bash
# 测试截图功能（需要手机已连接）
uv run python screenshot_test.py

# 测试 LLM 视觉能力（自动读取当前目录的 screenshot.png）
uv run python llm_test.py
```

## Policy 实验模块

`policy_expr/` 用于测试“截图观察 → 策略决策 → 执行动作 → 可选验证”的策略链路，适合对比不同 policy、验证多步 ReAct 表现。

### 运行模式

- `single-step`：默认模式，只执行一轮 ReAct。未指定 `--context` 时不读取历史；指定后会加载单用户对话上下文，只基于历史执行一步。
- `agent-loop`：单条用户目标内部的多步 ReAct。未指定 `--context` 时创建新 `PolicyContext`；指定后从该路径加载历史；每轮后手动确认是否继续。传 `--auto-continue` 时动作执行后自动进入下一轮截图验收，直到目标完成、失败或达到最大轮数。
- `dialog-loop`：单用户多轮自然语言对话模式。当前仅保留框架，运行时会提示 TODO，尚未完整实现。

每次启动都会创建独立运行目录：`logs/policy_expr/<mode>/<启动时间>/`。本次运行的截图、动作可视化、动作后截图和 `context.json` 都固定保存到该目录；`--context` 只用于指定要加载的历史 context 路径。

### 常用命令

```bash
# 单步策略测试
uv run python policy_expr/runner.py "打开微信"

# 带历史上下文的单步测试
uv run python policy_expr/runner.py "点通讯录" \
  --mode single-step \
  --context logs/policy_expr/single-step/20260505_120000/context.json

# 单目标内部多步 ReAct
uv run python policy_expr/runner.py "打开微信并进入通讯录" \
  --mode agent-loop

# 自动继续并限制循环轮数
uv run python policy_expr/runner.py "打开微信并进入通讯录" \
  --mode agent-loop \
  --auto-continue \
  --max-turns 8

# dialog-loop 目前仅提示 TODO
uv run python policy_expr/runner.py \
  --mode dialog-loop
```

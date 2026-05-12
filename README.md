# iphone-use

通过 MCP 协议控制 iPhone 的 AI Agent，核心模块在 `policy_expr/`。

## 环境配置

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
API_PROVIDER=modelscope
MODELSCOPE_API_KEY=your_api_key
MODELSCOPE_MODEL=Qwen/Qwen3.5-35B-A3B
```

## 运行

```bash
# 单步策略测试
uv run python policy_expr/runner.py "打开微信"

# 单目标多步 ReAct
uv run python policy_expr/runner.py "打开微信并进入通讯录" --mode agent-loop

# 自动继续并限制轮数
uv run python policy_expr/runner.py "打开微信并进入通讯录" \
  --mode agent-loop --auto-continue --max-turns 8

# 侦察模块 CLI
uv run python policy_expr/recon_cli.py
```

## 项目结构

- `policy_expr/` — 核心策略引擎（感知 → 决策 → 执行 → 验收）
- `llm/` — LLM 调用封装
- `models/` — 本地模型（图标检测等）
- `sck/` — SCStream 截图服务
- `bin/` — 预编译二进制
- `scripts/` — 测试脚本

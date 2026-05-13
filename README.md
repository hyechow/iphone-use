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

# 自学习（侦察当前页面 → 生成知识库）
uv run python -m policy_expr.recon_cli --app 微信

# BFS 多页探索，depth=N 递归探测发现的目标页面
uv run python -m policy_expr.recon_cli --app 微信 --depth 1

# 在线探测（截图 → probe，不生成知识库）
uv run python -m policy_expr.recon_cli --debug-parse

# 离线解析截图
uv run python -m policy_expr.recon_cli --debug-parse images/recon/

# 从侦察结果生成功能流程描述
uv run python -m policy_expr.recon_cli --debug-learn logs/recon/微信/聊天列表/recon_result.json

# 从侦察目录构建页面操作知识库
uv run python -m policy_expr.recon_cli --debug-knowledge logs/recon/微信/聊天列表
```

## 项目结构

- `policy_expr/` — 核心策略引擎（感知 → 决策 → 执行 → 验收）
- `policy_expr/recon_cli.py` — 自学习 CLI 入口
- `policy_expr/self_learning/` — 页面流描述 + 知识库抽象
- `knowledge/` — 按应用组织的页面操作知识库（skill markdown）
- `llm/` — LLM 调用封装
- `models/` — 本地模型（图标检测等）
- `sck/` — SCStream 截图服务

## 知识库构建流程

`--app` 模式自动完成完整流程：

1. **解析身份** — 截图 → LLM 识别页面签名
2. **去重检查** — 查找 `knowledge/{app}/` 下是否已有相同签名的 skill
3. **元素探测** — 逐个点击可交互元素，记录跳转结果 → `logs/recon/{app}/{page}/`
4. **功能描述** — 对每个元素生成点击后的跳转描述 → `page_flows.json`
5. **知识抽象** — LLM 合并同类、去除私有信息 → `knowledge/{app}/{page}.md`

`--depth N` 启用 BFS：探测完成后，依次点击每个可导航元素进入目标页面，递归执行上述流程，然后返回当前页面继续。
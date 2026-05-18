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

# DFS 多页探索，depth=N 决定可生成知识的页面层数
# depth=1：探测根页面所有子节点 → 可生成根页面的知识
# depth=2：再探测每个子节点的子节点 → 可生成根页面及其子页面的知识
uv run python -m policy_expr.recon_cli --app 微信 --depth 1 --sample 2

# 新增页面：手动导航到新页面后，加入已有应用的知识库
uv run python -m policy_expr.recon_cli --app 微信 --mode add --depth 1 --sample 2

# 更新页面：手动导航到目标页面后，重新探测指定页面（子页面仍走常规去重）
uv run python -m policy_expr.recon_cli --app 微信 --mode update --target "微信主界面，显示聊天列表和底部导航栏。"

# 在线探测（截图 → probe，不生成知识库）
uv run python -m policy_expr.recon_cli --debug-parse

# 离线解析截图
uv run python -m policy_expr.recon_cli --debug-parse images/recon/

# 从侦察结果生成功能流程描述
uv run python -m policy_expr.recon_cli --debug-learn logs/recon/微信/聊天列表/recon_result.json

# 从侦察目录构建页面操作知识库
uv run python -m policy_expr.recon_cli --debug-knowledge logs/recon/微信/聊天列表

# 截图数据采集（纯 DFS，不去重，用于 fingerprint/相似度分析）
uv run python scripts/collect_screenshot_dataset.py 微信 --depth 2
uv run python scripts/collect_screenshot_dataset.py 微信 --depth 2 --sample 3
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
4. **操作日志** — 每探测完一个页面，更新 `logs/recon/{app}/recon_log.json`（记录 trigger: initial/add/update）

### `--mode` 操作模式

| 模式 | 触发 trigger | 用途 |
|------|------------|------|
| 无（默认） | `initial` | 首次探索应用 |
| `add` | `add` | 手动导航到新页面后，加入已有应用 |
| `update` | `update` | 重新探测指定页面，需配合 `--target` 指定目录名 |

### `--depth N` 的含义

`--depth N` 启用 DFS 多层探索。**depth 决定可以生成知识的页面层数**，而非仅探索层数——因为生成某页面的知识需要先探测它的所有子节点。

| depth | 探测范围 | 可生成知识的页面 |
|-------|---------|---------------|
| 0 | 仅记录根页面，不探测 | 无 |
| 1 | 探测根页面的所有可交互元素（点击进入子页面后立刻返回，不再深入） | 根页面 |
| 2 | 在 depth=1 基础上，进一步探测每个子页面的元素 | 根页面 + 根页面的所有子页面 |
| N | 以此类推 | 从根页面到第 N-1 层的所有页面 |

> 最深一层（第 N 层）的页面只记录不探测，因此无法生成知识。
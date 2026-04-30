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

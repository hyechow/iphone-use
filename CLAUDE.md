# iphone-use

通过 MCP 协议控制 iPhone 的项目。使用 `mirroir-mcp` 作为 MCP server，提供截图等手机控制能力。

## 环境配置

```bash
# 安装依赖
brew tap jfarcand/tap
npx -y mirroir-mcp install

# 安装 Python 依赖
uv sync
```

## 运行

```bash
# 测试截图功能
uv run python screenshot_test.py
```

## 技术栈

- Python 3.11+
- `mcp` — MCP client，连接 mirroir-mcp server
- `anthropic` — Claude API
- `fastapi` + `uvicorn` — HTTP server
- `uv` — 包管理

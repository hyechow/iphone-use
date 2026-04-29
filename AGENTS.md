# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python 3.11+ iPhone automation assistant built around MCP.
Core agent logic lives in `agent/`: tools, runners, events, sessions, logging,
and MCP clients. `backend/` contains the FastAPI server and SSE API.
`frontend/index.html` is the browser UI served by the backend. `llm/` stores
provider configuration, and `scripts/` contains smoke tests and device/LLM
experiments. Visual references belong in `images/`; runtime logs belong in
`logs/` and are not source.

## Build, Test, and Development Commands

- `uv sync`: install the pinned dependencies from `pyproject.toml` and `uv.lock`.
- `uv run uvicorn backend.main:app --reload`: start the local web app at
  `http://localhost:8000`.
- `uv run python cli.py`: run the terminal REPL for interactive phone control.
- `uv run python scripts/screenshot_test.py`: verify mirroir-mcp screenshot access.
- `uv run python scripts/llm_test.py`: exercise the configured LLM provider.

Before device workflows, install mirroir-mcp with `brew tap jfarcand/tap` and
`npx -y mirroir-mcp install`.

## Coding Style & Naming Conventions

Use idiomatic Python with 4-space indentation, type hints for public interfaces,
and concise docstrings where behavior is not obvious. Prefer `snake_case` for
functions, variables, modules, and scripts; use `PascalCase` for classes. Keep
async boundaries explicit: backend routes and MCP calls stay `async`, while
synchronous CLI code stays isolated in `cli.py`. Do not commit generated logs,
cache files, or local screenshots unless they are required fixtures.

## Testing Guidelines

There is no formal pytest suite yet; validation is script-based under `scripts/`.
Name new checks `*_test.py` and make each script runnable with
`uv run python scripts/<name>_test.py`. Hardware-dependent tests should fail
clearly when the iPhone or MCP server is unavailable. For backend or agent
changes, run the relevant smoke script plus the affected web or CLI path.

## Commit & Pull Request Guidelines

Recent history uses short, imperative subjects such as `Add type_text tool` and
`Fix agent vision pipeline and add tap/home tools`. Follow that style: capitalize
the first word, describe the user-visible change, and keep the subject focused.
Pull requests should include a summary, commands run, required environment or
device setup, and screenshots or logs for UI/device behavior changes. Link issues
when available.

## Security & Configuration Tips

Keep provider secrets in local environment files or shell configuration, never in
source. Common variables include `API_PROVIDER`, `MODELSCOPE_API_KEY`,
`NVIDIA_API_KEY`, and provider-specific model names. Avoid checking in raw device
captures if they may contain personal data.

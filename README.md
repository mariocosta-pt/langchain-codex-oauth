# langchain-codex-oauth

Use the Codex models included with a **ChatGPT Plus/Pro** subscription inside **LangChain/LangGraph**.

This project authenticates locally via OpenAI OAuth (no `OPENAI_API_KEY`) and talks to the ChatGPT/Codex backend.

## What this is
- A dev-friendly adapter that makes Codex feel like a native-ish LangChain chat model.
- Local OAuth login (`langchain-codex-oauth auth login`) storing credentials under `~/.langchain-codex-oauth/`.
- Streaming support via SSE (`.stream()` yields chunks as they arrive).
- Async support via `.ainvoke()` / `.astream()`.
- Tool calling via `.bind_tools(...)` (useful for LangGraph agents).
- Streaming tool call chunks (`tool_call_chunks`) when available.
- Stop sequences (`stop=[...]`) for invoke/stream (best-effort).
- Response and usage metadata (`response_metadata`, `usage_metadata`) when available.

## What this is not
- Not for sharing accounts/subscriptions.
- Not intended for production or multi-user hosting.

## Requirements
- Python `>=3.10`
- Active ChatGPT Plus/Pro subscription with Codex access

## Install
```bash
python -m pip install langchain-codex-oauth
```

## Authenticate (one time)
```bash
langchain-codex-oauth auth login
# If port 1455 is busy or you’re on a remote machine:
langchain-codex-oauth auth login --manual
```

## Quickstart
```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_codex_oauth import ChatCodexOAuth

model = ChatCodexOAuth(model="gpt-5.2-codex")

messages = [
    SystemMessage(content="You are a concise assistant."),
    HumanMessage(content="Say hello and give a one-line coding tip."),
]

for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
```

Or run the examples (see `examples/README.md` for details):
```bash
python examples/langchain/hello.py
python examples/langchain/tools.py
python examples/langchain/async_hello.py
python examples/langchain/chatopenai_compatibility.py
python examples/langchain/flags_and_params.py
python examples/langchain/usage_and_metadata.py
python examples/langchain/tool_call_chunks.py

# Optional (requires installing langgraph in a separate venv)
python examples/langgraph/system_prompt_drift.py --mode strict
```

## Configuration knobs (ChatOpenAI-like)
`ChatCodexOAuth` supports a small set of common `ChatOpenAI` knobs:
- `timeout` (seconds) and `max_retries`
- `temperature` and `max_tokens` (best-effort passthrough)
- `stop=[...]` for `.invoke/.stream/.ainvoke/.astream` (best-effort)

Additional Codex-specific knob:
- `system_prompt_mode` (`"strict"` default, `"default"`, `"disabled"`) to reduce system-prompt drift on the consumer backend.

## Notes
- The Codex backend requires validated `instructions`. By default the library uses cached prompts, attempts GitHub fetch if missing, and falls back to bundled prompts (override with `LANGCHAIN_CODEX_OAUTH_INSTRUCTIONS_MODE`).
- The adapter aims to be OpenAI-like: tool schemas from `convert_to_openai_tool(...)` and `with_structured_output(...)` are supported.
- If you hit ChatGPT usage limits, the library normalizes some backend “usage limit” errors to HTTP 429 semantics.
- Usage counts and finish reasons are best-effort (the backend is not a stable public API).

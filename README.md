# langchain-codex-oauth

Use the Codex models included with a **ChatGPT Plus/Pro** subscription inside **LangChain/LangGraph**.

This project authenticates locally via OpenAI OAuth (no `OPENAI_API_KEY`) and talks to the ChatGPT/Codex backend.

## What this is
- A dev-friendly adapter that makes Codex feel like a native-ish LangChain chat model.
- Local OAuth login (`langchain-codex-oauth auth login`) storing credentials under `~/.langchain-codex-oauth/`.
- Streaming support via SSE (`.stream()` yields chunks as they arrive).
- Async support via `.ainvoke()` / `.astream()`.
- Tool calling via `.bind_tools(...)` (useful for LangGraph agents).

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

Or run the examples:
```bash
python examples/hello.py
python examples/tools.py
```

## Notes
- The Codex backend requires validated `instructions`. By default the library uses cached prompts, attempts GitHub fetch if missing, and falls back to bundled prompts (override with `LANGCHAIN_CODEX_OAUTH_INSTRUCTIONS_MODE`).
- If you hit ChatGPT usage limits, the library normalizes some backend “usage limit” errors to HTTP 429 semantics.

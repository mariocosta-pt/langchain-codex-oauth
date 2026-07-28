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

## Stability (v1)

- The Python API (`ChatCodexOAuth`, CLI commands, and message/tool wiring) is considered stable.
- The upstream backend is a ChatGPT consumer endpoint and may change without notice. If that happens, new releases may be required to restore compatibility.
- Recommendation: pin a minor version in real projects (e.g. `langchain-codex-oauth>=1.1,<1.2`) and upgrade when needed.

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

model = ChatCodexOAuth(model="gpt-5.6-sol")

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

## GPT-5.6 models

The adapter supports the current Codex OAuth model IDs directly:

| Model | Recommended use |
| --- | --- |
| `gpt-5.6-sol` | Complex planning, review, and the hardest agentic work |
| `gpt-5.6-terra` | Balanced implementation and everyday graph nodes |
| `gpt-5.6-luna` | Fast parsing, classification, and high-volume nodes |

`gpt-5.6-sol` is the default. Both `gpt-5.6-sol` and the provider-prefixed
`openai-codex/gpt-5.6-sol` form are accepted.

For example, a LangGraph workflow can choose a model and reasoning effort per
node:

```python
from langchain_codex_oauth import ChatCodexOAuth

planner = ChatCodexOAuth(
    model="gpt-5.6-sol",
    reasoning={"effort": "high"},
)
builder = ChatCodexOAuth(
    model="gpt-5.6-terra",
    reasoning={"effort": "medium"},
)
parser = ChatCodexOAuth(
    model="gpt-5.6-luna",
    reasoning={"effort": "low"},
)
```

## Configuration knobs (ChatOpenAI-like)

`ChatCodexOAuth` supports a small set of common `ChatOpenAI` knobs:

- `timeout` (seconds) and `max_retries`
- `temperature` and `max_tokens` (best-effort passthrough)
- `stop=[...]` for `.invoke/.stream/.ainvoke/.astream` (best-effort)
- `reasoning={"effort": ..., "summary": ...}` for OpenAI Responses-style reasoning controls
- `verbosity="low" | "medium" | "high"` as a ChatOpenAI-style alias for `text_verbosity`

Reasoning effort can also be supplied as a model suffix, inspired by Codex proxies:

```python
from langchain_codex_oauth import ChatCodexOAuth

# Sends upstream model="gpt-5.6-terra" with high effort and auto summary.
model = ChatCodexOAuth(
    model="gpt-5.6-terra-high",
    reasoning={"summary": "auto"},
    verbosity="medium",
)

# Equivalent explicit Responses-style configuration.
model = ChatCodexOAuth(
    model="gpt-5.6-terra",
    reasoning={"effort": "high", "summary": "auto"},
    verbosity="medium",
)
```

Supported effort suffixes/values are `none`, `minimal`, `low`, `medium`,
`high`, `xhigh`, and `max`. The GPT-5.6 models exposed by Codex currently list
`low`, `medium`, `high`, `xhigh`, and `max` reasoning levels.
Reasoning summary values are backend/model-dependent; common values are `auto`, `concise`, `detailed`, and `none`.
If summaries are returned, they are exposed under `AIMessage.response_metadata["reasoning"]`; encrypted reasoning content is not exposed, only marked as present.

Backward-compatible Codex-specific aliases are still supported:

- `reasoning_effort`, `reasoning_summary`
- `text_verbosity`
- `include` (defaults to `['reasoning.encrypted_content']` for reasoning-state continuity)
- `system_prompt_mode` (`"strict"` default, `"default"`, `"disabled"`) to reduce system-prompt drift on the consumer backend.

Note: Codex CLI Fast mode/service tier is separate from reasoning effort and is not exposed yet. Codex `ultra` is multi-agent orchestration rather than a normal Responses API reasoning effort, so this chat-model adapter does not expose it.

## RAG / Embeddings

`ChatCodexOAuth` is a chat model, not an embedding model.

- For RAG, you can pair it with local embeddings (e.g. Ollama `mxbai-embed-large`) or OpenAI embeddings.
- The critical rule is: the embedding model used to index documents must match the embedding model used to embed queries.
- See `examples/langchain/rag_chroma_ab.py` for an A/B harness (Chroma + persisted local DB).

## Notes

- The Codex backend requires validated `instructions`. By default the library uses cached prompts, attempts GitHub fetch if missing, and falls back to bundled prompts (override with `LANGCHAIN_CODEX_OAUTH_INSTRUCTIONS_MODE`).
- The adapter aims to be OpenAI-like: tool schemas from `convert_to_openai_tool(...)` and `with_structured_output(...)` are supported.
- If you hit ChatGPT usage limits, the library normalizes some backend “usage limit” errors to HTTP 429 semantics.
- Usage counts and finish reasons are best-effort (the backend is not a stable public API).

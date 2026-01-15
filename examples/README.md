# Examples

This directory contains runnable scripts for validating `langchain-codex-oauth`.

Before running any example, authenticate once:

```bash
langchain-codex-oauth auth login
```

## `examples/langchain/`
These examples use LangChain (`langchain-core`) only.

From the repo root:

```bash
python examples/langchain/hello.py
python examples/langchain/tools.py
python examples/langchain/async_hello.py
python examples/langchain/chatopenai_compatibility.py
python examples/langchain/flags_and_params.py
python examples/langchain/usage_and_metadata.py
python examples/langchain/tool_call_chunks.py
python examples/langchain/rag_chroma_ab.py
```

If you are running from a fresh environment:

```bash
python -m pip install -e ".[dev]"
```

Note: a couple of examples use `pydantic` for schema definitions. If your
environment does not already have it:

```bash
python -m pip install pydantic
```

## `examples/langchain/` — RAG (optional extra deps)
This RAG example uses Chroma as a local vector store and supports an A/B matrix:
- LLM: `--llm codex|openai`
- Embedder: `--embedder ollama|openai`

It persists the DB under `examples/output/` (gitignored).

Install extra deps in your example venv:

```bash
python -m pip install langchain-chroma chromadb
python -m pip install langchain-ollama
# Optional (for OpenAI embeddings or ChatOpenAI)
python -m pip install langchain-openai
```

Run (common combos):

```bash
# "Free-ish" dev flow: local embeddings + Codex via OAuth
python examples/langchain/rag_chroma_ab.py --llm codex --embedder ollama --ollama-embed-model mxbai-embed-large:latest

# Mixed flow: OpenAI embeddings + Codex via OAuth
python examples/langchain/rag_chroma_ab.py --llm codex --embedder openai --openai-embedding-model text-embedding-3-small

# Prod-like flow: OpenAI embeddings + ChatOpenAI
python examples/langchain/rag_chroma_ab.py --llm openai --embedder openai --openai-chat-model gpt-4o-mini
```

## `examples/langgraph/` (optional)
These examples validate LangGraph-specific workflows (e.g. system-prompt drift
mitigation). **LangGraph is intentionally not a dependency of this package.**

Recommended setup: create a separate virtualenv and install LangGraph manually.

From the repo root:

```bash
python -m venv examples/langgraph/.venv
source examples/langgraph/.venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install langgraph

python examples/langgraph/system_prompt_drift.py --mode strict
python examples/langgraph/travel_planner_graph.py --city Tokyo --budget 600 --mode strict
# Push it harder (budget must be close to cap, forces better tool/instruction adherence)
python examples/langgraph/travel_planner_graph.py --city Tokyo --budget 600 --mode strict --min-spend-ratio 0.95
```

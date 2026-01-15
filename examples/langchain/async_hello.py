"""Async hello example for `langchain-codex-oauth`.

Prereq (one-time login):
  `langchain-codex-oauth auth login`

Then run:
  `python examples/langchain/async_hello.py`

This uses native async `.ainvoke()` and `.astream()`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_codex_oauth import ChatCodexOAuth


async def main() -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    messages = [
        SystemMessage(content="You are a concise assistant."),
        HumanMessage(content="Say hello and give a one-line coding tip."),
    ]

    print("Async streaming response:\n")
    async for chunk in model.astream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")

    msg = await model.ainvoke([HumanMessage(content="Reply with exactly: OK")])
    print("Async invoke response:")
    print(msg.content)


if __name__ == "__main__":
    asyncio.run(main())

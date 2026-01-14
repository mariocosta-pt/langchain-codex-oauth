"""Hello example for `langchain-codex-oauth`.

Prereq (one-time login):
  `langchain-codex-oauth auth login`

Then run:
  `python examples/hello.py`

Notes:
- The first model call may fetch Codex instructions from GitHub.
- This example works without installing the package (it adds repo root to PYTHONPATH).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_codex_oauth import ChatCodexOAuth


def main() -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    messages = [
        SystemMessage(content="You are a concise, helpful assistant."),
        HumanMessage(content="Say hello, then give me a one-line coding tip."),
    ]

    print("Streaming response:\n")
    for chunk in model.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()

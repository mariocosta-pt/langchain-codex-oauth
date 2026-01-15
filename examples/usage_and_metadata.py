"""v0.5 parity example (response_metadata + usage_metadata).

Prereq (one-time login):
  `langchain-codex-oauth auth login`

Then run:
  `python examples/usage_and_metadata.py`

Note: usage fields are best-effort; depending on backend/model, usage may be
missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage

from langchain_codex_oauth import ChatCodexOAuth


def main() -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    msg = model.invoke([HumanMessage(content="Say hello in one short sentence.")])
    print("content:", msg.content)
    print("response_metadata:", msg.response_metadata)
    print("usage_metadata:", msg.usage_metadata)

    chunks = list(model.stream([HumanMessage(content="Stream a short hello.")]))
    text = "".join(str(c.content) for c in chunks)
    last = chunks[-1]

    print("\nstreamed text:", text)
    print("last.response_metadata:", last.response_metadata)
    print("last.usage_metadata:", last.usage_metadata)


if __name__ == "__main__":
    main()

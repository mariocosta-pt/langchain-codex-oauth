"""Config knobs example (timeout/retries/temperature/max_tokens/stop).

Prereq (one-time login):
  `langchain-codex-oauth auth login`

Then run:
  `python examples/flags_and_params.py`

This demonstrates ChatOpenAI-like knobs supported by ChatCodexOAuth.
Note: some generation knobs are best-effort passthroughs; if the Codex backend
rejects a parameter, the request retries without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage

from langchain_codex_oauth import ChatCodexOAuth


def main() -> None:
    model = ChatCodexOAuth(
        model="gpt-5.2-codex",
        timeout=30.0,
        max_retries=1,
        temperature=0.2,
        max_tokens=60,
    )

    print("Invoke with stop sequence (truncate at END):")
    msg = model.invoke(
        [HumanMessage(content="Write: alpha END omega")],
        stop=["END"],
    )
    print(repr(msg.content))
    print("contains END:", "END" in msg.content)

    print("\nStream with stop sequence (truncate at END):")
    out = "".join(
        str(chunk.content)
        for chunk in model.stream(
            [HumanMessage(content="Stream: alpha END omega")],
            stop=["END"],
        )
    )
    print(repr(out))
    print("contains END:", "END" in out)


if __name__ == "__main__":
    main()

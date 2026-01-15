"""v0.6 parity example (stream tool_call_chunks).

Prereq (one-time login):
  `langchain-codex-oauth auth login`

Then run:
  `python examples/langchain/tool_call_chunks.py`

Depending on the backend/model, tool args may stream in chunks or may arrive as a
single final tool call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class Answer(BaseModel):
    answer: str


def main() -> None:
    from langchain_codex_oauth import ChatCodexOAuth

    base = ChatCodexOAuth(model="gpt-5.2-codex")
    tool_schema = convert_to_openai_tool(Answer)
    model = base.bind_tools([tool_schema], tool_choice="any")

    prompt = "Call the Answer tool with answer='hello'."

    print("Streaming...\n")
    chunks = []
    for chunk in model.stream([HumanMessage(content=prompt)]):
        if getattr(chunk, "tool_call_chunks", None):
            for tc in chunk.tool_call_chunks:
                print("tool_call_chunk:", tc)
        if getattr(chunk, "tool_calls", None):
            print("tool_calls:", chunk.tool_calls)
        chunks.append(chunk)

    last = chunks[-1]
    print("\nfinal response_metadata:", last.response_metadata)
    print("final usage_metadata:", last.usage_metadata)

    # Best-effort: show reconstructed args from streamed chunks.
    assembled: dict[str, str] = {}
    for c in chunks:
        for tc in getattr(c, "tool_call_chunks", []) or []:
            call_id = tc.get("id")
            delta = tc.get("args")
            if isinstance(call_id, str) and isinstance(delta, str):
                assembled[call_id] = assembled.get(call_id, "") + delta

    if assembled:
        print("\nassembled args deltas:")
        for call_id, args in assembled.items():
            print(call_id, "->", args)
            try:
                print("parsed:", json.loads(args))
            except Exception:
                pass


if __name__ == "__main__":
    main()

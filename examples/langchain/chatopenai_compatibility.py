"""v0.4 parity example (stop + dict tool schema + structured output).

Prereq (one-time login):
  `langchain-codex-oauth auth login`

Then run:
  `python examples/langchain/chatopenai_compatibility.py`

This demonstrates compatibility features designed to make switching to ChatOpenAI
in production as painless as possible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from langchain_codex_oauth import ChatCodexOAuth


class Answer(BaseModel):
    answer: str


def main() -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    print("Stop sequences (truncate at STOP):")
    msg = model.invoke(
        [HumanMessage(content="Write: hello STOP world")],
        stop=["STOP"],
    )
    print(msg.content)

    print("\nTool schema dict compatibility (convert_to_openai_tool output):")
    schema_dict = convert_to_openai_tool(Answer)
    tool_model = model.bind_tools([schema_dict], tool_choice="any")
    first = tool_model.invoke(
        [HumanMessage(content="Return a JSON object with key 'answer' and a greeting.")]
    )
    print(first.tool_calls)

    print("\nStructured output (with_structured_output):")
    structured = model.with_structured_output(Answer, include_raw=True)
    out = structured.invoke([HumanMessage(content="Return a short greeting")])

    if isinstance(out, dict):
        print("parsed:", out.get("parsed"))
    else:
        print("parsed:", out)


if __name__ == "__main__":
    main()

"""Tool-calling example for `langchain-codex-oauth`.

Prereq (one-time login):
  `langchain-codex-oauth auth login`

Then run:
  `python examples/langchain/tools.py`

This demonstrates the same basic loop used by LangGraph agents:
1) Model requests tools via `AIMessage.tool_calls`
2) You execute tools and return `ToolMessage` results
3) Model continues with the tool outputs in context
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_codex_oauth import ChatCodexOAuth


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""

    return a + b


def main() -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex").bind_tools([add])

    messages = [HumanMessage(content="What is 123 + 456?")]

    first = model.invoke(messages)
    if first.content:
        print(first.content)

    print("Tool calls:")
    print(first.tool_calls)

    if not first.tool_calls:
        print("No tool calls requested.")
        return

    tool_messages: list[ToolMessage] = []
    for call in first.tool_calls:
        result = add.invoke(call)
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    final = model.invoke([*messages, first, *tool_messages])
    print("\nResponse:")
    print(final.content)


if __name__ == "__main__":
    main()

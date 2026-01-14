from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function


def convert_tools(
    tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
) -> list[dict[str, Any]]:
    """Convert LangChain tool-like objects to OpenAI tool schema dicts."""

    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = convert_to_openai_function(tool)
        if not isinstance(function, dict):
            raise TypeError("Tool conversion produced a non-dict schema")

        # Codex backend expects Responses-style function tools:
        # {"type":"function","name":...,"description":...,"parameters":...}
        converted.append({"type": "function", **function})
    return converted


def normalize_tool_choice(tool_choice: Any | None) -> Any | None:
    """Normalize tool_choice to OpenAI-compatible formats.

    Accepts common LangChain conventions:
    - None: omit tool_choice
    - "auto": let the model decide
    - "any": treated like "auto"
    - "required": must call a tool
    - "<tool_name>": force a specific function tool
    """

    if tool_choice is None:
        return None

    if isinstance(tool_choice, dict):
        return tool_choice

    if not isinstance(tool_choice, str):
        return tool_choice

    value = tool_choice.strip()
    lowered = value.lower()

    if lowered == "any":
        return "auto"
    if lowered in {"auto", "required"}:
        return lowered

    return {"type": "function", "name": value}

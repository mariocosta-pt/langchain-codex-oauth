from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.utils.function_calling import convert_to_openai_function


def _as_dict(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return None


def convert_tools(
    tools: Sequence[Any],
) -> list[dict[str, Any]]:
    """Convert tool-like objects into Codex backend tool schema.

    The ChatGPT/Codex backend expects Responses-style function tools:
    `{ "type": "function", "name": ..., "description": ..., "parameters": ... }`

    For parity with `ChatOpenAI`, accept dict tools in multiple common formats:
    - OpenAI tool schema: `{ "type": "function", "function": { ... } }`
    - OpenAI function schema: `{ "name": ..., "parameters": ... }`
    - Responses function schema: `{ "type": "function", "name": ..., ... }`
    """

    converted: list[dict[str, Any]] = []
    for tool in tools:
        tool_dict = _as_dict(tool)
        if tool_dict is not None:
            # OpenAI tool schema: {"type":"function","function":{...}}
            if tool_dict.get("type") == "function" and isinstance(
                tool_dict.get("function"), dict
            ):
                function = tool_dict["function"]
                converted.append({"type": "function", **function})
                continue

            # Responses-style function schema: {"type":"function","name":...}
            if tool_dict.get("type") == "function" and isinstance(
                tool_dict.get("name"), str
            ):
                converted.append(tool_dict)
                continue

            # OpenAI function schema: {"name":...,"parameters":...}
            if isinstance(tool_dict.get("name"), str) and isinstance(
                tool_dict.get("parameters"), dict
            ):
                converted.append({"type": "function", **tool_dict})
                continue

        function = convert_to_openai_function(tool)  # type: ignore[arg-type]
        if not isinstance(function, dict):
            raise TypeError("Tool conversion produced a non-dict schema")

        converted.append({"type": "function", **function})

    return converted


def normalize_tool_choice(tool_choice: Any | None) -> Any | None:
    """Normalize tool_choice to OpenAI-like formats.

    The Codex backend uses a Responses-style `tool_choice`.

    Accepted inputs:
    - None
    - "auto" | "none" | "required" | "any" (mapped)
    - "<tool_name>" (forced)
    - OpenAI chat format dict: {"type":"function","function":{"name":...}}
    - Responses format dict: {"type":"function","name":...}
    """

    if tool_choice is None:
        return None

    choice_dict = _as_dict(tool_choice)
    if choice_dict is not None:
        # OpenAI chat format dict
        if choice_dict.get("type") == "function" and isinstance(
            choice_dict.get("function"), dict
        ):
            name = choice_dict["function"].get("name")
            if isinstance(name, str) and name:
                return {"type": "function", "name": name}

        # Responses format dict
        if choice_dict.get("type") == "function" and isinstance(
            choice_dict.get("name"), str
        ):
            return choice_dict

        return choice_dict

    if not isinstance(tool_choice, str):
        return tool_choice

    value = tool_choice.strip()
    lowered = value.lower()

    if lowered == "any":
        # `with_structured_output` uses tool_choice="any" and expects a tool call.
        return "required"

    if lowered in {"auto", "none", "required"}:
        return lowered

    return {"type": "function", "name": value}

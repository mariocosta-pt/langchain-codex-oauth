from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict


class ToolCall(TypedDict):
    name: str
    args: dict[str, Any]
    id: str | None
    type: NotRequired[Literal["tool_call"]]


class InvalidToolCall(TypedDict):
    type: Literal["invalid_tool_call"]
    id: str | None
    name: str | None
    args: str | None
    error: str | None


@dataclass(frozen=True)
class ParsedAssistantMessage:
    content: str
    tool_calls: list[ToolCall]
    invalid_tool_calls: list[InvalidToolCall]


def parse_assistant_message(response: object) -> ParsedAssistantMessage:
    """Parse a Codex Responses-style `response` into assistant text + tool calls.

    This is intentionally tolerant: the ChatGPT/Codex backend is not a stable API.
    """

    if not isinstance(response, dict):
        return ParsedAssistantMessage(
            content=str(response) if response is not None else "",
            tool_calls=[],
            invalid_tool_calls=[],
        )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    invalid_tool_calls: list[InvalidToolCall] = []

    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")

            if item_type == "message":
                content = item.get("content")
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        block_type = block.get("type")
                        if block_type in {"output_text", "text"} and isinstance(
                            block.get("text"), str
                        ):
                            text_parts.append(block["text"])

            if item_type == "function_call":
                call_id = item.get("call_id") or item.get("id")
                name = item.get("name")
                arguments = item.get("arguments")

                call_id_str = call_id if isinstance(call_id, str) else None
                name_str = name if isinstance(name, str) else None

                if not name_str:
                    continue

                if isinstance(arguments, dict):
                    tool_calls.append(
                        {
                            "type": "tool_call",
                            "id": call_id_str,
                            "name": name_str,
                            "args": arguments,
                        }
                    )
                    continue

                if isinstance(arguments, str):
                    try:
                        parsed_args = json.loads(arguments)
                        if not isinstance(parsed_args, dict):
                            raise ValueError("arguments must be a JSON object")
                        tool_calls.append(
                            {
                                "type": "tool_call",
                                "id": call_id_str,
                                "name": name_str,
                                "args": parsed_args,
                            }
                        )
                    except Exception as exc:
                        invalid_tool_calls.append(
                            {
                                "type": "invalid_tool_call",
                                "id": call_id_str,
                                "name": name_str,
                                "args": arguments,
                                "error": str(exc),
                            }
                        )
                    continue

                invalid_tool_calls.append(
                    {
                        "type": "invalid_tool_call",
                        "id": call_id_str,
                        "name": name_str,
                        "args": None,
                        "error": "missing tool call arguments",
                    }
                )

    # Fallbacks
    if not text_parts:
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            text_parts.append(output_text)

    return ParsedAssistantMessage(
        content="".join(text_parts),
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )

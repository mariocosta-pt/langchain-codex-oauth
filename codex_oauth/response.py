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


@dataclass(frozen=True)
class CompletionResult:
    parsed: ParsedAssistantMessage
    response: object


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


def extract_response_metadata(response: object) -> dict[str, Any]:
    """Extract OpenAI-like response metadata from a Responses-style object."""

    if not isinstance(response, dict):
        return {}

    metadata: dict[str, Any] = {}

    response_id = response.get("id")
    if isinstance(response_id, str) and response_id:
        metadata["id"] = response_id

    model = response.get("model")
    if isinstance(model, str) and model:
        metadata["model"] = model

    status = response.get("status")
    if isinstance(status, str) and status:
        metadata["status"] = status

    created_at = response.get("created_at")
    if isinstance(created_at, (int, float)):
        metadata["created_at"] = int(created_at)

    # Best-effort finish_reason.
    finish_reason = response.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        metadata["finish_reason"] = finish_reason
    else:
        output = response.get("output")
        has_tool_calls = False
        if isinstance(output, list):
            has_tool_calls = any(
                isinstance(item, dict) and item.get("type") == "function_call"
                for item in output
            )

        incomplete = response.get("incomplete_details")
        if isinstance(incomplete, dict):
            reason = incomplete.get("reason")
            if isinstance(reason, str) and reason:
                metadata["finish_reason"] = "length" if "token" in reason else reason
        elif has_tool_calls:
            metadata["finish_reason"] = "tool_calls"
        elif status in {"completed", "done"}:
            metadata["finish_reason"] = "stop"

    return metadata


def extract_usage_metadata(response: object) -> dict[str, Any] | None:
    """Extract token usage when available.

    Returned dict is shaped like LangChain `UsageMetadata`:
    `{input_tokens, output_tokens, total_tokens}`.
    """

    if not isinstance(response, dict):
        return None

    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None

    def _as_int(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    input_tokens = _as_int(usage.get("input_tokens"))
    if input_tokens is None:
        input_tokens = _as_int(usage.get("prompt_tokens"))

    output_tokens = _as_int(usage.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _as_int(usage.get("completion_tokens"))

    total_tokens = _as_int(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    if input_tokens is None or output_tokens is None or total_tokens is None:
        return None

    result: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    return result

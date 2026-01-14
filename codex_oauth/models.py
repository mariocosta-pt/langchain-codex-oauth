from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

Role = Literal["developer", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """Minimal message type for plain-chat usage."""

    role: Role
    content: str


class InputText(TypedDict):
    type: Literal["input_text"]
    text: str


class InputMessageItem(TypedDict):
    type: Literal["message"]
    role: Role
    content: list[InputText]


class InputFunctionCallItem(TypedDict):
    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str


class InputFunctionCallOutputItem(TypedDict):
    type: Literal["function_call_output"]
    call_id: str
    output: str


InputItem = InputMessageItem | InputFunctionCallItem | InputFunctionCallOutputItem


def message_item(role: Role, text: str) -> InputMessageItem:
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }


def function_call_item(
    call_id: str, name: str, args: dict[str, Any] | str
) -> InputFunctionCallItem:
    if isinstance(args, str):
        arguments = args
    else:
        arguments = json.dumps(args, separators=(",", ":"))
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def function_call_output_item(call_id: str, output: Any) -> InputFunctionCallOutputItem:
    output_text = output if isinstance(output, str) else json.dumps(output)
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output_text,
    }


def messages_to_input(messages: list[ChatMessage]) -> list[InputItem]:
    return [message_item(message.role, message.content) for message in messages]


def normalize_model(model: str) -> str:
    model_id = model.split("/", 1)[1] if "/" in model else model
    return model_id.strip()

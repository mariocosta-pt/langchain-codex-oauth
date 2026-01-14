from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

Role = Literal["developer", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class InputText(TypedDict):
    type: Literal["input_text"]
    text: str


class InputMessageItem(TypedDict):
    type: Literal["message"]
    role: Role
    content: list[InputText]


def messages_to_input(messages: list[ChatMessage]) -> list[InputMessageItem]:
    return [
        {
            "type": "message",
            "role": message.role,
            "content": [{"type": "input_text", "text": message.content}],
        }
        for message in messages
    ]


def normalize_model(model: str) -> str:
    model_id = model.split("/", 1)[1] if "/" in model else model
    return model_id.strip()

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from typing import Any


def iter_sse_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Parse SSE lines into JSON events.

    Only `data:` fields are processed; other SSE fields are ignored.
    """

    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                if payload.strip() == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event
            continue

        if line.startswith(":"):
            continue

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        payload = "\n".join(data_lines)
        if payload.strip() != "[DONE]":
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                return
            if isinstance(event, dict):
                yield event


async def aiter_sse_events(lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
    """Async variant of `iter_sse_events`."""

    data_lines: list[str] = []
    async for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                if payload.strip() == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event
            continue

        if line.startswith(":"):
            continue

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        payload = "\n".join(data_lines)
        if payload.strip() != "[DONE]":
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                return
            if isinstance(event, dict):
                yield event


def is_terminal_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("type") or "")
    return event_type in {"response.done", "response.completed"}


def extract_text_delta(event: dict[str, Any]) -> str | None:
    """Best-effort extraction of user-visible text deltas.

    The Codex backend emits many `.delta` event types (including reasoning). For
    ChatOpenAI parity we only surface output-text deltas.
    """

    event_type = str(event.get("type") or "")

    # OpenAI Responses-style text streaming.
    # Example: {"type": "response.output_text.delta", "delta": "hi"}
    if event_type.endswith("output_text.delta"):
        if isinstance(event.get("delta"), str):
            return event["delta"]
        if isinstance(event.get("text"), str):
            return event["text"]

    # Some backends use a generic `...text.delta` with a `text` field.
    if event_type.endswith("text.delta") and isinstance(event.get("text"), str):
        return event["text"]

    return None

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
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


def is_terminal_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("type") or "")
    return event_type in {"response.done", "response.completed"}


def extract_text_delta(event: dict[str, Any]) -> str | None:
    """Best-effort extraction of text deltas from response events."""

    if isinstance(event.get("delta"), str):
        return event["delta"]

    event_type = str(event.get("type") or "")
    if event_type.endswith(".delta") and isinstance(event.get("text"), str):
        return event["text"]

    return None

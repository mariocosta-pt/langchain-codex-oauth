import asyncio
from typing import Any

from langchain_core.messages import HumanMessage

from langchain_codex_oauth import ChatCodexOAuth


def test_stream_emits_tool_call_chunks(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    terminal_response = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "Answer",
                "arguments": '{"answer": "hi"}',
            }
        ],
        "status": "completed",
    }

    def _fake_stream_events(**_: Any):
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "call_id": "call_123", "name": "Answer"},
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "call_id": "call_123",
            "delta": '{"answer": ',
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "call_id": "call_123",
            "delta": '"hi"}',
        }
        yield {"type": "response.done", "response": terminal_response}

    monkeypatch.setattr(model._client, "stream_events", _fake_stream_events)

    chunks = list(model.stream([HumanMessage(content="hi")]))

    delta_tool_chunks = [
        c
        for c in chunks
        if getattr(c, "tool_call_chunks", None)
        and getattr(c, "chunk_position", None) != "last"
    ]
    assert len(delta_tool_chunks) == 2

    first = delta_tool_chunks[0].tool_call_chunks[0]
    assert first["type"] == "tool_call_chunk"
    assert first["id"] == "call_123"
    assert first["name"] == "Answer"
    assert first["index"] == 0

    last = chunks[-1]
    tool_calls = getattr(last, "tool_calls", None)
    assert isinstance(tool_calls, list)
    assert tool_calls
    assert tool_calls[0]["id"] == "call_123"


def test_astream_emits_tool_call_chunks(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    terminal_response = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "Answer",
                "arguments": '{"answer": "hi"}',
            }
        ],
        "status": "completed",
    }

    async def _fake_astream_events(**_: Any):
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "call_id": "call_abc", "name": "Answer"},
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "call_id": "call_abc",
            "delta": '{"answer": ',
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "call_id": "call_abc",
            "delta": '"hi"}',
        }
        yield {"type": "response.done", "response": terminal_response}

    monkeypatch.setattr(model._async_client, "astream_events", _fake_astream_events)

    async def _run() -> list[Any]:
        out = []
        async for chunk in model.astream([HumanMessage(content="hi")]):
            out.append(chunk)
        return out

    chunks = asyncio.run(_run())

    delta_tool_chunks = [
        c
        for c in chunks
        if getattr(c, "tool_call_chunks", None)
        and getattr(c, "chunk_position", None) != "last"
    ]
    assert len(delta_tool_chunks) == 2
    assert delta_tool_chunks[0].tool_call_chunks[0]["id"] == "call_abc"

    last = chunks[-1]
    tool_calls = getattr(last, "tool_calls", None)
    assert isinstance(tool_calls, list)
    assert tool_calls
    assert tool_calls[0]["id"] == "call_abc"

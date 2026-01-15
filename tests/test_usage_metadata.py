import asyncio
from typing import Any

from langchain_core.messages import HumanMessage

from codex_oauth.response import CompletionResult, ParsedAssistantMessage
from langchain_codex_oauth import ChatCodexOAuth


def test_invoke_attaches_response_metadata_and_usage(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    response = {
        "id": "resp_test_123",
        "model": "gpt-5.2-codex",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }

    def _fake_complete_with_response(**_: Any) -> CompletionResult:
        parsed = ParsedAssistantMessage(
            content="hello", tool_calls=[], invalid_tool_calls=[]
        )
        return CompletionResult(parsed=parsed, response=response)

    monkeypatch.setattr(
        model._client, "complete_with_response", _fake_complete_with_response
    )

    msg = model.invoke([HumanMessage(content="hi")])
    assert msg.content == "hello"
    assert msg.response_metadata.get("id") == "resp_test_123"
    assert msg.response_metadata.get("model") == "gpt-5.2-codex"
    assert msg.response_metadata.get("finish_reason") == "stop"
    assert msg.usage_metadata == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }


def test_invoke_maps_incomplete_to_length_finish_reason(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    response = {
        "id": "resp_incomplete",
        "model": "gpt-5.2-codex",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    }

    def _fake_complete_with_response(**_: Any) -> CompletionResult:
        parsed = ParsedAssistantMessage(
            content="", tool_calls=[], invalid_tool_calls=[]
        )
        return CompletionResult(parsed=parsed, response=response)

    monkeypatch.setattr(
        model._client, "complete_with_response", _fake_complete_with_response
    )

    msg = model.invoke([HumanMessage(content="hi")])
    assert msg.response_metadata.get("finish_reason") == "length"


def test_stream_attaches_metadata_on_last_chunk(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    response = {
        "id": "resp_stream_1",
        "model": "gpt-5.2-codex",
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }

    def _fake_stream_events(**_: Any):
        yield {"type": "response.output_text.delta", "delta": "he"}
        yield {"type": "response.output_text.delta", "delta": "llo"}
        yield {"type": "response.done", "response": response}

    monkeypatch.setattr(model._client, "stream_events", _fake_stream_events)

    chunks = list(model.stream([HumanMessage(content="hi")]))
    assert "".join(str(c.content) for c in chunks) == "hello"

    last = chunks[-1]
    assert last.response_metadata.get("id") == "resp_stream_1"
    assert last.usage_metadata == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }


def test_ainvoke_attaches_response_metadata_and_usage(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    response = {
        "id": "resp_async_1",
        "model": "gpt-5.2-codex",
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }

    async def _fake_acomplete_with_response(**_: Any) -> CompletionResult:
        parsed = ParsedAssistantMessage(
            content="hello", tool_calls=[], invalid_tool_calls=[]
        )
        return CompletionResult(parsed=parsed, response=response)

    monkeypatch.setattr(
        model._async_client, "acomplete_with_response", _fake_acomplete_with_response
    )

    msg = asyncio.run(model.ainvoke([HumanMessage(content="hi")]))
    assert msg.response_metadata.get("id") == "resp_async_1"
    assert msg.usage_metadata == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }

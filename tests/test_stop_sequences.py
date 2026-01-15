import asyncio
from typing import Any

from langchain_core.messages import HumanMessage

from codex_oauth.response import ParsedAssistantMessage
from langchain_codex_oauth import ChatCodexOAuth


def test_stop_sequences_truncate_invoke(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    def _fake_complete(**_: Any) -> ParsedAssistantMessage:
        return ParsedAssistantMessage(
            content="hello STOP world", tool_calls=[], invalid_tool_calls=[]
        )

    monkeypatch.setattr(model._client, "complete", _fake_complete)

    msg = model.invoke([HumanMessage(content="hi")], stop=["STOP"])
    assert msg.content == "hello "


def test_stop_sequences_truncate_stream(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    def _fake_stream_events(**_: Any):
        yield {"type": "response.output_text.delta", "delta": "hello "}
        yield {"type": "response.output_text.delta", "delta": "ST"}
        yield {"type": "response.output_text.delta", "delta": "OP world"}
        yield {"type": "response.done", "response": {"output": []}}

    monkeypatch.setattr(model._client, "stream_events", _fake_stream_events)

    out = "".join(
        chunk.content
        for chunk in model.stream([HumanMessage(content="hi")], stop=["STOP"])
    )
    assert out == "hello "


def test_stop_sequences_truncate_ainvoke(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    async def _fake_acomplete(**_: Any) -> ParsedAssistantMessage:
        return ParsedAssistantMessage(
            content="hello STOP world", tool_calls=[], invalid_tool_calls=[]
        )

    monkeypatch.setattr(model._async_client, "acomplete", _fake_acomplete)

    msg = asyncio.run(model.ainvoke([HumanMessage(content="hi")], stop=["STOP"]))
    assert msg.content == "hello "


def test_stop_sequences_truncate_astream(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex")

    async def _fake_astream_events(**_: Any):
        yield {"type": "response.output_text.delta", "delta": "hello "}
        yield {"type": "response.output_text.delta", "delta": "ST"}
        yield {"type": "response.output_text.delta", "delta": "OP world"}
        yield {"type": "response.done", "response": {"output": []}}

    monkeypatch.setattr(model._async_client, "astream_events", _fake_astream_events)

    async def _run() -> str:
        chunks: list[str] = []
        async for chunk in model.astream([HumanMessage(content="hi")], stop=["STOP"]):
            if chunk.content:
                chunks.append(chunk.content)
        return "".join(chunks)

    out = asyncio.run(_run())
    assert out == "hello "

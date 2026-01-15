import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from codex_oauth.response import CompletionResult, ParsedAssistantMessage
from langchain_codex_oauth import ChatCodexOAuth


def test_system_prompt_mode_strict_attaches_extra_instructions(
    monkeypatch: Any,
) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex", system_prompt_mode="strict")

    captured: dict[str, Any] = {}

    def _fake_complete_with_response(**kwargs: Any) -> CompletionResult:
        captured.update(kwargs)
        parsed = ParsedAssistantMessage(
            content="ok", tool_calls=[], invalid_tool_calls=[]
        )
        return CompletionResult(
            parsed=parsed, response={"output": [], "status": "completed"}
        )

    monkeypatch.setattr(
        model._client, "complete_with_response", _fake_complete_with_response
    )

    msg = model.invoke(
        [SystemMessage(content="You are a router."), HumanMessage(content="hi")]
    )
    assert msg.content == "ok"

    input_items = captured.get("input_items")
    assert isinstance(input_items, list)
    assert input_items[0]["role"] == "developer"
    assert "router" in input_items[0]["content"][0]["text"].lower()

    extra = captured.get("extra_instructions")
    assert isinstance(extra, str)
    assert "router" in extra.lower()


def test_system_prompt_mode_default_does_not_attach_extra_instructions(
    monkeypatch: Any,
) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex", system_prompt_mode="default")

    captured: dict[str, Any] = {}

    def _fake_complete_with_response(**kwargs: Any) -> CompletionResult:
        captured.update(kwargs)
        parsed = ParsedAssistantMessage(
            content="ok", tool_calls=[], invalid_tool_calls=[]
        )
        return CompletionResult(
            parsed=parsed, response={"output": [], "status": "completed"}
        )

    monkeypatch.setattr(
        model._client, "complete_with_response", _fake_complete_with_response
    )

    model.invoke(
        [SystemMessage(content="You are a router."), HumanMessage(content="hi")]
    )

    input_items = captured.get("input_items")
    assert isinstance(input_items, list)
    assert input_items[0]["role"] == "developer"
    assert input_items[0]["content"][0]["text"] == "You are a router."

    assert captured.get("extra_instructions") is None


def test_system_prompt_mode_disabled_skips_system_prompts(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex", system_prompt_mode="disabled")

    captured: dict[str, Any] = {}

    def _fake_complete_with_response(**kwargs: Any) -> CompletionResult:
        captured.update(kwargs)
        parsed = ParsedAssistantMessage(
            content="ok", tool_calls=[], invalid_tool_calls=[]
        )
        return CompletionResult(
            parsed=parsed, response={"output": [], "status": "completed"}
        )

    monkeypatch.setattr(
        model._client, "complete_with_response", _fake_complete_with_response
    )

    model.invoke(
        [SystemMessage(content="You are a router."), HumanMessage(content="hi")]
    )

    input_items = captured.get("input_items")
    assert isinstance(input_items, list)
    assert input_items[0]["role"] == "user"

    assert captured.get("extra_instructions") is None


def test_stream_passes_extra_instructions_in_strict_mode(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex", system_prompt_mode="strict")

    captured: dict[str, Any] = {}

    def _fake_stream_events(**kwargs: Any):
        captured.update(kwargs)
        yield {
            "type": "response.done",
            "response": {"output": [], "status": "completed"},
        }

    monkeypatch.setattr(model._client, "stream_events", _fake_stream_events)

    chunks = list(
        model.stream(
            [SystemMessage(content="You are a router."), HumanMessage(content="hi")]
        )
    )
    assert chunks

    extra = captured.get("extra_instructions")
    assert isinstance(extra, str)
    assert "router" in extra.lower()


def test_astream_passes_extra_instructions_in_strict_mode(monkeypatch: Any) -> None:
    model = ChatCodexOAuth(model="gpt-5.2-codex", system_prompt_mode="strict")

    captured: dict[str, Any] = {}

    async def _fake_astream_events(**kwargs: Any):
        captured.update(kwargs)
        yield {
            "type": "response.done",
            "response": {"output": [], "status": "completed"},
        }

    monkeypatch.setattr(model._async_client, "astream_events", _fake_astream_events)

    async def _run() -> list[Any]:
        out = []
        async for chunk in model.astream(
            [SystemMessage(content="You are a router."), HumanMessage(content="hi")]
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(_run())
    assert chunks

    extra = captured.get("extra_instructions")
    assert isinstance(extra, str)
    assert "router" in extra.lower()

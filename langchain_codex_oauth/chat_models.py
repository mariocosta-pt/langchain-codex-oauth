from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, cast

from codex_oauth.client import AsyncCodexClient, CodexClient
from codex_oauth.models import (
    InputItem,
    function_call_item,
    function_call_output_item,
    message_item,
)
from codex_oauth.response import ParsedAssistantMessage, parse_assistant_message
from codex_oauth.sse import extract_text_delta, is_terminal_event
from codex_oauth.store import AuthStore
from langchain_codex_oauth.tooling import convert_tools, normalize_tool_choice

try:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        ToolMessage,
    )
    from langchain_core.messages.tool import ToolCall
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "langchain-core is required. Install with: pip install langchain-codex-oauth"
    ) from exc


def _ensure_tool_call_ids(tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
    normalized: list[ToolCall] = []
    for call in tool_calls:
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{uuid.uuid4().hex}"
        updated = {**call, "id": call_id, "type": "tool_call"}
        normalized.append(cast(ToolCall, updated))
    return normalized


def _to_input_items(messages: list[BaseMessage]) -> list[InputItem]:
    items: list[InputItem] = []

    for message in messages:
        if message.type in {"system", "developer"}:
            items.append(message_item("developer", str(message.content)))
            continue

        if message.type in {"human", "user"}:
            items.append(message_item("user", str(message.content)))
            continue

        if isinstance(message, ToolMessage) or message.type == "tool":
            tool_call_id = getattr(message, "tool_call_id", None)
            if isinstance(tool_call_id, str) and tool_call_id:
                items.append(function_call_output_item(tool_call_id, message.content))
            continue

        # Assistant message
        assistant_text = str(message.content) if message.content else ""
        if assistant_text:
            items.append(message_item("assistant", assistant_text))

        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            for tool_call in _ensure_tool_call_ids(tool_calls):
                name = tool_call.get("name")
                args = tool_call.get("args")
                call_id = tool_call.get("id")

                if not isinstance(name, str) or not name:
                    continue
                if not isinstance(call_id, str) or not call_id:
                    continue

                if not isinstance(args, dict):
                    args = {}

                items.append(function_call_item(call_id=call_id, name=name, args=args))

    return items


class ChatCodexOAuth(BaseChatModel):
    model: str = "gpt-5.2-codex"
    reasoning_effort: str | None = "medium"
    reasoning_summary: str | None = "auto"
    text_verbosity: str | None = "medium"
    include: list[str] | None = ["reasoning.encrypted_content"]

    def __init__(
        self,
        *,
        model: str | None = None,
        auth_store: AuthStore | None = None,
        reasoning_effort: str | None = "medium",
        reasoning_summary: str | None = "auto",
        text_verbosity: str | None = "medium",
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model or self.model
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.text_verbosity = text_verbosity
        self.include = include

        store = auth_store or AuthStore()
        self._client = CodexClient(auth_store=store)
        self._async_client = AsyncCodexClient(auth_store=store)

    @property
    def _llm_type(self) -> str:
        return "codex_oauth"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_summary": self.reasoning_summary,
            "text_verbosity": self.text_verbosity,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        openai_tools = convert_tools(tools)
        normalized_choice = normalize_tool_choice(tool_choice)
        return self.bind(tools=openai_tools, tool_choice=normalized_choice, **kwargs)

    def _complete(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any | None,
    ) -> ParsedAssistantMessage:
        return self._client.complete(
            input_items=_to_input_items(messages),
            model=self.model,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            text_verbosity=self.text_verbosity,
            include=self.include,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            raise ValueError("stop sequences are not supported yet")

        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        parsed = self._complete(
            messages,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice,
        )

        tool_calls = _ensure_tool_call_ids(parsed.tool_calls)

        message = AIMessage(
            content=parsed.content,
            tool_calls=tool_calls,
            invalid_tool_calls=parsed.invalid_tool_calls,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if stop:
            raise ValueError("stop sequences are not supported yet")

        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        input_items = _to_input_items(messages)

        for event in self._client.stream_events(
            input_items=input_items,
            model=self.model,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            text_verbosity=self.text_verbosity,
            include=self.include,
        ):
            if is_terminal_event(event):
                parsed = parse_assistant_message(event.get("response"))
                tool_calls = _ensure_tool_call_ids(parsed.tool_calls)

                if tool_calls or parsed.invalid_tool_calls:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            tool_calls=tool_calls,
                            invalid_tool_calls=parsed.invalid_tool_calls,
                            chunk_position="last",
                        )
                    )
                return

            delta = extract_text_delta(event)
            if delta:
                if run_manager:
                    run_manager.on_llm_new_token(delta)
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            raise ValueError("stop sequences are not supported yet")

        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        parsed = await self._async_client.acomplete(
            input_items=_to_input_items(messages),
            model=self.model,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            text_verbosity=self.text_verbosity,
            include=self.include,
        )

        tool_calls = _ensure_tool_call_ids(parsed.tool_calls)
        message = AIMessage(
            content=parsed.content,
            tool_calls=tool_calls,
            invalid_tool_calls=parsed.invalid_tool_calls,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if stop:
            raise ValueError("stop sequences are not supported yet")

        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        input_items = _to_input_items(messages)

        async for event in self._async_client.astream_events(
            input_items=input_items,
            model=self.model,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            text_verbosity=self.text_verbosity,
            include=self.include,
        ):
            if is_terminal_event(event):
                parsed = parse_assistant_message(event.get("response"))
                tool_calls = _ensure_tool_call_ids(parsed.tool_calls)

                if tool_calls or parsed.invalid_tool_calls:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            tool_calls=tool_calls,
                            invalid_tool_calls=parsed.invalid_tool_calls,
                            chunk_position="last",
                        )
                    )
                return

            delta = extract_text_delta(event)
            if delta:
                if run_manager:
                    await run_manager.on_llm_new_token(delta)
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))

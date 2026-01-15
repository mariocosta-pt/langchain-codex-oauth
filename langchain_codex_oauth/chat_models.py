from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, cast

from codex_oauth.client import CODEX_BASE_URL, AsyncCodexClient, CodexClient
from codex_oauth.env import get_env_float, get_env_int, get_env_str
from codex_oauth.models import (
    InputItem,
    function_call_item,
    function_call_output_item,
    message_item,
)
from codex_oauth.response import (
    CompletionResult,
    ParsedAssistantMessage,
    extract_response_metadata,
    extract_usage_metadata,
    parse_assistant_message,
)
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


def _truncate_at_stop(text: str, stop: list[str] | None) -> str:
    if not stop:
        return text

    earliest: int | None = None
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx

    return text[:earliest] if earliest is not None else text


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
    reasoning_summary: str | None = None
    text_verbosity: str | None = "medium"
    include: list[str] | None = ["reasoning.encrypted_content"]

    # Common ChatOpenAI-style knobs (best-effort).
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    max_retries: int | None = None
    base_url: str | None = None

    def __init__(
        self,
        *,
        model: str | None = None,
        auth_store: AuthStore | None = None,
        reasoning_effort: str | None = "medium",
        reasoning_summary: str | None = None,
        text_verbosity: str | None = "medium",
        include: list[str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model or self.model
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.text_verbosity = text_verbosity
        self.include = include

        env_base_url = get_env_str("LANGCHAIN_CODEX_OAUTH_BASE_URL")
        env_timeout_s = get_env_float("LANGCHAIN_CODEX_OAUTH_TIMEOUT_S")
        env_max_retries = get_env_int("LANGCHAIN_CODEX_OAUTH_MAX_RETRIES")
        env_temperature = get_env_float("LANGCHAIN_CODEX_OAUTH_TEMPERATURE")
        env_max_tokens = get_env_int("LANGCHAIN_CODEX_OAUTH_MAX_TOKENS")

        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url

        self.temperature = temperature
        self.max_tokens = max_tokens

        store = auth_store or AuthStore()

        resolved_base_url = base_url or env_base_url or CODEX_BASE_URL
        resolved_timeout_s = (
            float(timeout)
            if timeout is not None
            else (env_timeout_s if env_timeout_s is not None else 60.0)
        )
        resolved_max_retries = (
            int(max_retries)
            if max_retries is not None
            else (env_max_retries if env_max_retries is not None else 2)
        )

        if self.temperature is None:
            self.temperature = env_temperature
        if self.max_tokens is None:
            self.max_tokens = env_max_tokens

        self._client = CodexClient(
            auth_store=store,
            base_url=resolved_base_url,
            timeout_s=resolved_timeout_s,
            max_retries=resolved_max_retries,
        )
        self._async_client = AsyncCodexClient(
            auth_store=store,
            base_url=resolved_base_url,
            timeout_s=resolved_timeout_s,
            max_retries=resolved_max_retries,
        )

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
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        openai_tools = convert_tools(tools)
        normalized_choice = normalize_tool_choice(tool_choice)
        return self.bind(tools=openai_tools, tool_choice=normalized_choice, **kwargs)

    def _complete_with_response(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any | None,
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> CompletionResult:
        return self._client.complete_with_response(
            input_items=_to_input_items(messages),
            model=self.model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            text_verbosity=self.text_verbosity,
            include=self.include,
        )

    def _complete(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any | None,
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> ParsedAssistantMessage:
        return self._complete_with_response(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ).parsed

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        temperature = kwargs.get("temperature", getattr(self, "temperature", None))
        max_tokens = kwargs.get("max_tokens", getattr(self, "max_tokens", None))

        result = self._complete_with_response(
            messages,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice,
            temperature=temperature if isinstance(temperature, (int, float)) else None,
            max_output_tokens=max_tokens if isinstance(max_tokens, int) else None,
        )

        parsed = result.parsed
        response_metadata = extract_response_metadata(result.response)
        usage_metadata = extract_usage_metadata(result.response)

        content = _truncate_at_stop(parsed.content, stop)
        tool_calls = _ensure_tool_call_ids(parsed.tool_calls)

        message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            invalid_tool_calls=parsed.invalid_tool_calls,
            response_metadata=response_metadata,
            usage_metadata=usage_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        temperature = kwargs.get("temperature", getattr(self, "temperature", None))
        max_tokens = kwargs.get("max_tokens", getattr(self, "max_tokens", None))

        input_items = _to_input_items(messages)

        stop_sequences = [s for s in (stop or []) if s]
        max_stop_len = max((len(s) for s in stop_sequences), default=0)
        buffer = ""
        stopped = False

        for event in self._client.stream_events(
            input_items=input_items,
            model=self.model,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice,
            temperature=temperature if isinstance(temperature, (int, float)) else None,
            max_output_tokens=max_tokens if isinstance(max_tokens, int) else None,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            text_verbosity=self.text_verbosity,
            include=self.include,
        ):
            if is_terminal_event(event):
                if not stopped and buffer:
                    if run_manager:
                        run_manager.on_llm_new_token(buffer)
                    yield ChatGenerationChunk(message=AIMessageChunk(content=buffer))

                raw_response = event.get("response")
                parsed = parse_assistant_message(raw_response)
                tool_calls = _ensure_tool_call_ids(parsed.tool_calls)

                response_metadata = extract_response_metadata(raw_response)
                usage_metadata = extract_usage_metadata(raw_response)

                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_calls=tool_calls,
                        invalid_tool_calls=parsed.invalid_tool_calls,
                        response_metadata=response_metadata,
                        usage_metadata=usage_metadata,
                        chunk_position="last",
                    )
                )
                return

            delta = extract_text_delta(event)
            if not delta or stopped:
                continue

            buffer += delta

            if stop_sequences:
                # If any stop sequence is present, emit up to its start and stop.
                earliest: int | None = None
                for s in stop_sequences:
                    idx = buffer.find(s)
                    if idx != -1 and (earliest is None or idx < earliest):
                        earliest = idx

                if earliest is not None:
                    emit_text = buffer[:earliest]
                    if emit_text:
                        if run_manager:
                            run_manager.on_llm_new_token(emit_text)
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(content=emit_text)
                        )
                    stopped = True
                    buffer = ""
                    continue

                # Emit only the safe prefix, keeping a lookbehind to match stop tokens
                # that may span chunk boundaries.
                if max_stop_len > 1:
                    safe_len = max(0, len(buffer) - (max_stop_len - 1))
                else:
                    safe_len = len(buffer)

                emit_text = buffer[:safe_len]
                buffer = buffer[safe_len:]
                if emit_text:
                    if run_manager:
                        run_manager.on_llm_new_token(emit_text)
                    yield ChatGenerationChunk(message=AIMessageChunk(content=emit_text))
            else:
                # No stop sequences: emit immediately.
                if run_manager:
                    run_manager.on_llm_new_token(buffer)
                yield ChatGenerationChunk(message=AIMessageChunk(content=buffer))
                buffer = ""

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        temperature = kwargs.get("temperature", getattr(self, "temperature", None))
        max_tokens = kwargs.get("max_tokens", getattr(self, "max_tokens", None))

        result = await self._async_client.acomplete_with_response(
            input_items=_to_input_items(messages),
            model=self.model,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice,
            temperature=temperature if isinstance(temperature, (int, float)) else None,
            max_output_tokens=max_tokens if isinstance(max_tokens, int) else None,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            text_verbosity=self.text_verbosity,
            include=self.include,
        )

        parsed = result.parsed
        response_metadata = extract_response_metadata(result.response)
        usage_metadata = extract_usage_metadata(result.response)

        content = _truncate_at_stop(parsed.content, stop)
        tool_calls = _ensure_tool_call_ids(parsed.tool_calls)

        message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            invalid_tool_calls=parsed.invalid_tool_calls,
            response_metadata=response_metadata,
            usage_metadata=usage_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        temperature = kwargs.get("temperature", getattr(self, "temperature", None))
        max_tokens = kwargs.get("max_tokens", getattr(self, "max_tokens", None))

        input_items = _to_input_items(messages)

        stop_sequences = [s for s in (stop or []) if s]
        max_stop_len = max((len(s) for s in stop_sequences), default=0)
        buffer = ""
        stopped = False

        async for event in self._async_client.astream_events(
            input_items=input_items,
            model=self.model,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice,
            temperature=temperature if isinstance(temperature, (int, float)) else None,
            max_output_tokens=max_tokens if isinstance(max_tokens, int) else None,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            text_verbosity=self.text_verbosity,
            include=self.include,
        ):
            if is_terminal_event(event):
                if not stopped and buffer:
                    if run_manager:
                        await run_manager.on_llm_new_token(buffer)
                    yield ChatGenerationChunk(message=AIMessageChunk(content=buffer))

                raw_response = event.get("response")
                parsed = parse_assistant_message(raw_response)
                tool_calls = _ensure_tool_call_ids(parsed.tool_calls)

                response_metadata = extract_response_metadata(raw_response)
                usage_metadata = extract_usage_metadata(raw_response)

                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_calls=tool_calls,
                        invalid_tool_calls=parsed.invalid_tool_calls,
                        response_metadata=response_metadata,
                        usage_metadata=usage_metadata,
                        chunk_position="last",
                    )
                )
                return

            delta = extract_text_delta(event)
            if not delta or stopped:
                continue

            buffer += delta

            if stop_sequences:
                earliest: int | None = None
                for s in stop_sequences:
                    idx = buffer.find(s)
                    if idx != -1 and (earliest is None or idx < earliest):
                        earliest = idx

                if earliest is not None:
                    emit_text = buffer[:earliest]
                    if emit_text:
                        if run_manager:
                            await run_manager.on_llm_new_token(emit_text)
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(content=emit_text)
                        )
                    stopped = True
                    buffer = ""
                    continue

                if max_stop_len > 1:
                    safe_len = max(0, len(buffer) - (max_stop_len - 1))
                else:
                    safe_len = len(buffer)

                emit_text = buffer[:safe_len]
                buffer = buffer[safe_len:]
                if emit_text:
                    if run_manager:
                        await run_manager.on_llm_new_token(emit_text)
                    yield ChatGenerationChunk(message=AIMessageChunk(content=emit_text))
            else:
                if run_manager:
                    await run_manager.on_llm_new_token(buffer)
                yield ChatGenerationChunk(message=AIMessageChunk(content=buffer))
                buffer = ""

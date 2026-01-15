from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Literal, cast

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


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _extract_tool_call_item_added(
    event: dict[str, Any],
) -> tuple[int, str, str | None] | None:
    """Extract tool call identity from `response.output_item.added` events."""

    if event.get("type") != "response.output_item.added":
        return None

    output_index = _coerce_int(event.get("output_index"))
    if output_index is None:
        return None

    item = event.get("item")
    if not isinstance(item, dict):
        return None

    if item.get("type") != "function_call":
        return None

    call_id = item.get("call_id") or item.get("id") or event.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        return None

    name = item.get("name")
    name_str = name if isinstance(name, str) and name else None

    return output_index, call_id, name_str


def _extract_tool_call_args_delta(event: dict[str, Any]) -> tuple[int, str, str] | None:
    """Extract tool call argument deltas from streaming events."""

    if event.get("type") != "response.function_call_arguments.delta":
        return None

    output_index = _coerce_int(event.get("output_index"))
    if output_index is None:
        return None

    call_id = event.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        return None

    delta = event.get("delta")
    if not isinstance(delta, str) or not delta:
        return None

    return output_index, call_id, delta


def _tool_call_chunk(
    *,
    call_id: str,
    name: str | None,
    args_delta: str,
    index: int,
) -> Any:
    return {
        "type": "tool_call_chunk",
        "id": call_id,
        "name": name,
        "args": args_delta,
        "index": index,
    }


SystemPromptMode = Literal["strict", "default", "disabled"]


def _extract_system_texts(messages: list[BaseMessage]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if message.type in {"system", "developer"}:
            text = str(message.content)
            if text:
                texts.append(text)
    return texts


def _format_system_prompt_strict(texts: list[str]) -> str:
    joined = "\n\n".join(t for t in texts if t)
    return f"System instructions (highest priority):\n{joined}".strip()


def _build_extra_instructions(texts: list[str]) -> str | None:
    if not texts:
        return None

    joined = "\n\n".join(t for t in texts if t)
    if not joined:
        return None

    max_chars = 4000
    if len(joined) > max_chars:
        joined = joined[:max_chars].rstrip() + "..."

    return (
        "### Conversation system prompt\n"
        "Treat the following system instructions as highest priority.\n\n"
        f"{joined}\n\n"
        "### End conversation system prompt"
    )


def _to_input_items(
    messages: list[BaseMessage], *, system_prompt_mode: SystemPromptMode = "default"
) -> list[InputItem]:
    items: list[InputItem] = []

    mode = system_prompt_mode

    if mode == "strict":
        system_texts = _extract_system_texts(messages)
        if system_texts:
            items.append(
                message_item("developer", _format_system_prompt_strict(system_texts))
            )
        messages_to_process = [
            m for m in messages if m.type not in {"system", "developer"}
        ]
    elif mode == "disabled":
        messages_to_process = [
            m for m in messages if m.type not in {"system", "developer"}
        ]
    else:
        messages_to_process = list(messages)

    for message in messages_to_process:
        if mode == "default" and message.type in {"system", "developer"}:
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

    # Drift mitigation; default to strict for reliability.
    system_prompt_mode: SystemPromptMode = "strict"

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
        system_prompt_mode: SystemPromptMode = "strict",
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

        self.system_prompt_mode = system_prompt_mode

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
            "system_prompt_mode": self.system_prompt_mode,
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
        system_texts = (
            _extract_system_texts(messages)
            if self.system_prompt_mode == "strict"
            else []
        )
        extra_instructions = (
            _build_extra_instructions(system_texts)
            if self.system_prompt_mode == "strict"
            else None
        )

        return self._client.complete_with_response(
            input_items=_to_input_items(
                messages, system_prompt_mode=self.system_prompt_mode
            ),
            model=self.model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            text_verbosity=self.text_verbosity,
            include=self.include,
            extra_instructions=extra_instructions,
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

        system_texts = (
            _extract_system_texts(messages)
            if self.system_prompt_mode == "strict"
            else []
        )
        extra_instructions = (
            _build_extra_instructions(system_texts)
            if self.system_prompt_mode == "strict"
            else None
        )

        input_items = _to_input_items(
            messages, system_prompt_mode=self.system_prompt_mode
        )

        stop_sequences = [s for s in (stop or []) if s]
        max_stop_len = max((len(s) for s in stop_sequences), default=0)
        buffer = ""
        stopped = False

        tool_call_name_by_id: dict[str, str | None] = {}
        tool_call_index_by_id: dict[str, int] = {}

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
            extra_instructions=extra_instructions,
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

            added = _extract_tool_call_item_added(event)
            if added and not stopped:
                output_index, call_id, name = added
                tool_call_name_by_id[call_id] = name
                tool_call_index_by_id[call_id] = output_index
                continue

            args_delta = _extract_tool_call_args_delta(event)
            if args_delta and not stopped:
                output_index, call_id, delta_text = args_delta

                if call_id not in tool_call_index_by_id:
                    tool_call_index_by_id[call_id] = output_index
                if call_id not in tool_call_name_by_id:
                    tool_call_name_by_id[call_id] = None

                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            _tool_call_chunk(
                                call_id=call_id,
                                name=tool_call_name_by_id.get(call_id),
                                args_delta=delta_text,
                                index=tool_call_index_by_id[call_id],
                            )
                        ],
                    )
                )
                continue

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

        system_texts = (
            _extract_system_texts(messages)
            if self.system_prompt_mode == "strict"
            else []
        )
        extra_instructions = (
            _build_extra_instructions(system_texts)
            if self.system_prompt_mode == "strict"
            else None
        )

        result = await self._async_client.acomplete_with_response(
            input_items=_to_input_items(
                messages, system_prompt_mode=self.system_prompt_mode
            ),
            model=self.model,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice,
            temperature=temperature if isinstance(temperature, (int, float)) else None,
            max_output_tokens=max_tokens if isinstance(max_tokens, int) else None,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            text_verbosity=self.text_verbosity,
            include=self.include,
            extra_instructions=extra_instructions,
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

        system_texts = (
            _extract_system_texts(messages)
            if self.system_prompt_mode == "strict"
            else []
        )
        extra_instructions = (
            _build_extra_instructions(system_texts)
            if self.system_prompt_mode == "strict"
            else None
        )

        input_items = _to_input_items(
            messages, system_prompt_mode=self.system_prompt_mode
        )

        stop_sequences = [s for s in (stop or []) if s]
        max_stop_len = max((len(s) for s in stop_sequences), default=0)
        buffer = ""
        stopped = False

        tool_call_name_by_id: dict[str, str | None] = {}
        tool_call_index_by_id: dict[str, int] = {}

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
            extra_instructions=extra_instructions,
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

            added = _extract_tool_call_item_added(event)
            if added and not stopped:
                output_index, call_id, name = added
                tool_call_name_by_id[call_id] = name
                tool_call_index_by_id[call_id] = output_index
                continue

            args_delta = _extract_tool_call_args_delta(event)
            if args_delta and not stopped:
                output_index, call_id, delta_text = args_delta

                if call_id not in tool_call_index_by_id:
                    tool_call_index_by_id[call_id] = output_index
                if call_id not in tool_call_name_by_id:
                    tool_call_name_by_id[call_id] = None

                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            _tool_call_chunk(
                                call_id=call_id,
                                name=tool_call_name_by_id.get(call_id),
                                args_delta=delta_text,
                                index=tool_call_index_by_id[call_id],
                            )
                        ],
                    )
                )
                continue

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

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from codex_oauth.client import CodexClient
from codex_oauth.models import ChatMessage
from codex_oauth.store import AuthStore

try:
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "langchain-core is required. Install with: pip install langchain-codex-oauth"
    ) from exc


def _to_codex_messages(messages: list[BaseMessage]) -> list[ChatMessage]:
    codex_messages: list[ChatMessage] = []
    for message in messages:
        role: str
        if message.type in {"system", "developer"}:
            role = "developer"
        elif message.type in {"human", "user"}:
            role = "user"
        else:
            role = "assistant"
        codex_messages.append(ChatMessage(role=role, content=str(message.content)))
    return codex_messages


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
        self._client = CodexClient(auth_store=auth_store or AuthStore())

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

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            raise ValueError("stop sequences are not supported yet")
        text = self._client.chat(
            messages=_to_codex_messages(messages),
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            text_verbosity=self.text_verbosity,
            include=self.include,
        )
        message = AIMessage(content=text)
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
        for delta in self._client.stream_chat(
            messages=_to_codex_messages(messages),
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            text_verbosity=self.text_verbosity,
            include=self.include,
        ):
            if run_manager:
                run_manager.on_llm_new_token(delta)
            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))

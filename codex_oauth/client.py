from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from codex_oauth.auth import (
    arefresh_access_token,
    decode_jwt_payload,
    extract_chatgpt_account_id,
    refresh_access_token,
)
from codex_oauth.exceptions import CodexAPIError, NotAuthenticatedError
from codex_oauth.instructions import aget_codex_instructions, get_codex_instructions
from codex_oauth.models import (
    ChatMessage,
    InputItem,
    messages_to_input,
    normalize_model,
)
from codex_oauth.response import (
    CompletionResult,
    ParsedAssistantMessage,
    parse_assistant_message,
)
from codex_oauth.sse import (
    aiter_sse_events,
    extract_text_delta,
    is_terminal_event,
    iter_sse_events,
)
from codex_oauth.store import AuthStore, OAuthCredentials

CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_RESPONSES_PATH = "/codex/responses"

DEFAULT_INCLUDE = ["reasoning.encrypted_content"]


def _backoff_s(attempt: int) -> float:
    base = min(8.0, 0.5 * (2**attempt))
    return base * (1.0 + random.random() * 0.1)


def _is_retryable_status(status_code: int | None) -> bool:
    return status_code in {429, 500, 502, 503, 504}


class CodexClient:
    def __init__(
        self,
        auth_store: AuthStore | None = None,
        *,
        base_url: str = CODEX_BASE_URL,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._store = auth_store or AuthStore()
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    def _load_valid_credentials(self, http: httpx.Client) -> OAuthCredentials:
        creds = self._store.load()
        now_ms = int(time.time() * 1000)
        if creds.expires > now_ms:
            return creds

        refreshed = refresh_access_token(refresh_token=creds.refresh, http=http)
        payload = decode_jwt_payload(refreshed.access)
        if not payload:
            raise NotAuthenticatedError(
                "Token refresh succeeded but token is invalid; re-login required."
            )
        account_id = extract_chatgpt_account_id(payload)
        if not account_id:
            raise NotAuthenticatedError(
                "Failed to derive account id from refreshed token; re-login required."
            )

        new_creds = OAuthCredentials(
            access=refreshed.access,
            refresh=refreshed.refresh,
            expires=refreshed.expires_at_ms,
            account_id=account_id,
        )
        self._store.save(new_creds)
        return new_creds

    @staticmethod
    def _headers(creds: OAuthCredentials) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {creds.access}",
            "chatgpt-account-id": creds.account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "Accept": "text/event-stream",
        }

    def stream_events(
        self,
        *,
        input_items: list[InputItem],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        text_verbosity: str | None = None,
        include: list[str] | None = None,
        extra_instructions: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        request_body: dict[str, Any] = {
            "model": normalize_model(model),
            "store": False,
            "stream": True,
            "input": input_items,
            "include": DEFAULT_INCLUDE if include is None else include,
        }

        if tools is not None:
            request_body["tools"] = tools
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice
        if temperature is not None:
            request_body["temperature"] = temperature
        if max_output_tokens is not None:
            request_body["max_output_tokens"] = max_output_tokens

        if reasoning_effort or reasoning_summary:
            request_body["reasoning"] = {
                **({"effort": reasoning_effort} if reasoning_effort else {}),
                **({"summary": reasoning_summary} if reasoning_summary else {}),
            }
        if text_verbosity:
            request_body["text"] = {"verbosity": text_verbosity}

        url = f"{self._base_url}{CODEX_RESPONSES_PATH}"
        with httpx.Client(timeout=self._timeout_s) as http:
            creds = self._load_valid_credentials(http)
            base_instructions = get_codex_instructions(
                http, model=request_body["model"]
            )
            instructions_extra_removed = False
            if extra_instructions:
                request_body["instructions"] = (
                    f"{base_instructions}\n\n{extra_instructions}".strip()
                )
            else:
                request_body["instructions"] = base_instructions

            tool_choice_removed = False
            temperature_removed = False
            max_output_tokens_removed = False
            attempt = 0
            while True:
                try:
                    with http.stream(
                        "POST",
                        url,
                        headers=self._headers(creds),
                        json=request_body,
                    ) as response:
                        if response.status_code >= 400:
                            try:
                                response.read()
                            except Exception:
                                pass

                            err = self._to_api_error(response)

                            if (
                                not instructions_extra_removed
                                and extra_instructions
                                and err.status_code == 400
                                and "instruction" in str(err).lower()
                            ):
                                request_body["instructions"] = base_instructions
                                instructions_extra_removed = True
                                continue

                            if (
                                not tool_choice_removed
                                and tool_choice is not None
                                and err.status_code == 400
                                and "tool_choice" in str(err).lower()
                            ):
                                request_body.pop("tool_choice", None)
                                tool_choice_removed = True
                                continue

                            if (
                                not temperature_removed
                                and temperature is not None
                                and err.status_code == 400
                                and "temperature" in str(err).lower()
                            ):
                                request_body.pop("temperature", None)
                                temperature_removed = True
                                continue

                            if (
                                not max_output_tokens_removed
                                and max_output_tokens is not None
                                and err.status_code == 400
                                and any(
                                    key in str(err).lower()
                                    for key in ("max_output_tokens", "max_tokens")
                                )
                            ):
                                request_body.pop("max_output_tokens", None)
                                max_output_tokens_removed = True
                                continue

                            if (
                                _is_retryable_status(err.status_code)
                                and attempt < self._max_retries
                            ):
                                time.sleep(_backoff_s(attempt))
                                attempt += 1
                                continue

                            raise err

                        yield from iter_sse_events(response.iter_lines())
                        return

                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt < self._max_retries:
                        time.sleep(_backoff_s(attempt))
                        attempt += 1
                        continue
                    raise CodexAPIError(
                        "Network error calling Codex backend", status_code=None
                    ) from exc

    def complete_with_response(
        self,
        *,
        input_items: list[InputItem],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        text_verbosity: str | None = None,
        include: list[str] | None = None,
        extra_instructions: str | None = None,
    ) -> CompletionResult:
        last_response: object | None = None
        for event in self.stream_events(
            input_items=input_items,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            text_verbosity=text_verbosity,
            include=include,
            extra_instructions=extra_instructions,
        ):
            if is_terminal_event(event):
                last_response = event.get("response")
                break

        return CompletionResult(
            parsed=parse_assistant_message(last_response),
            response=last_response,
        )

    def complete(
        self,
        *,
        input_items: list[InputItem],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        text_verbosity: str | None = None,
        include: list[str] | None = None,
        extra_instructions: str | None = None,
    ) -> ParsedAssistantMessage:
        return self.complete_with_response(
            input_items=input_items,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            text_verbosity=text_verbosity,
            include=include,
            extra_instructions=extra_instructions,
        ).parsed

    # Backwards-compatible helpers for plain chat.
    def stream_chat(
        self,
        *,
        messages: list[ChatMessage],
        model: str,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        text_verbosity: str | None = None,
        include: list[str] | None = None,
    ) -> Iterator[str]:
        for event in self.stream_events(
            input_items=messages_to_input(messages),
            model=model,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            text_verbosity=text_verbosity,
            include=include,
        ):
            if is_terminal_event(event):
                return
            delta = extract_text_delta(event)
            if delta:
                yield delta

    def chat(
        self,
        *,
        messages: list[ChatMessage],
        model: str,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        text_verbosity: str | None = None,
        include: list[str] | None = None,
    ) -> str:
        return self.complete(
            input_items=messages_to_input(messages),
            model=model,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            text_verbosity=text_verbosity,
            include=include,
        ).content

    @staticmethod
    def _to_api_error(response: httpx.Response) -> CodexAPIError:
        status = response.status_code
        text = ""
        try:
            text = response.text
        except Exception:
            text = ""

        safe_excerpt = text[:1000]
        message = f"Codex backend request failed (HTTP {status})."

        code: str | None = None
        detail: str | None = None
        try:
            parsed = json.loads(text) if text else None
            if isinstance(parsed, dict):
                err = parsed.get("error")
                if isinstance(err, dict):
                    raw_code = err.get("code") or err.get("type")
                    code = raw_code if isinstance(raw_code, str) else None

                raw_detail = parsed.get("detail")
                detail = raw_detail if isinstance(raw_detail, str) else None

            if code:
                message = f"Codex backend request failed (HTTP {status}, {code})."
        except Exception:
            pass

        # The ChatGPT subscription backend sometimes returns usage limits as 404.
        haystack = f"{code or ''} {detail or ''} {text}".lower()
        is_usage_limit = any(
            token in haystack
            for token in (
                "usage_limit_reached",
                "usage_not_included",
                "rate_limit_exceeded",
                "usage limit",
                "too many requests",
            )
        )
        if status == 404 and is_usage_limit:
            status = 429
            message = (
                "Codex usage limit reached for your ChatGPT subscription "
                "(treated as HTTP 429)."
            )

        if safe_excerpt:
            message = f"{message} Response excerpt: {safe_excerpt}"

        return CodexAPIError(message, status_code=status)


class AsyncCodexClient:
    def __init__(
        self,
        auth_store: AuthStore | None = None,
        *,
        base_url: str = CODEX_BASE_URL,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._store = auth_store or AuthStore()
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    async def _load_valid_credentials(
        self, http: httpx.AsyncClient
    ) -> OAuthCredentials:
        creds = self._store.load()
        now_ms = int(time.time() * 1000)
        if creds.expires > now_ms:
            return creds

        refreshed = await arefresh_access_token(refresh_token=creds.refresh, http=http)
        payload = decode_jwt_payload(refreshed.access)
        if not payload:
            raise NotAuthenticatedError(
                "Token refresh succeeded but token is invalid; re-login required."
            )
        account_id = extract_chatgpt_account_id(payload)
        if not account_id:
            raise NotAuthenticatedError(
                "Failed to derive account id from refreshed token; re-login required."
            )

        new_creds = OAuthCredentials(
            access=refreshed.access,
            refresh=refreshed.refresh,
            expires=refreshed.expires_at_ms,
            account_id=account_id,
        )
        self._store.save(new_creds)
        return new_creds

    @staticmethod
    def _headers(creds: OAuthCredentials) -> dict[str, str]:
        return CodexClient._headers(creds)

    async def astream_events(
        self,
        *,
        input_items: list[InputItem],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        text_verbosity: str | None = None,
        include: list[str] | None = None,
        extra_instructions: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        request_body: dict[str, Any] = {
            "model": normalize_model(model),
            "store": False,
            "stream": True,
            "input": input_items,
            "include": DEFAULT_INCLUDE if include is None else include,
        }

        if tools is not None:
            request_body["tools"] = tools
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice
        if temperature is not None:
            request_body["temperature"] = temperature
        if max_output_tokens is not None:
            request_body["max_output_tokens"] = max_output_tokens

        if reasoning_effort or reasoning_summary:
            request_body["reasoning"] = {
                **({"effort": reasoning_effort} if reasoning_effort else {}),
                **({"summary": reasoning_summary} if reasoning_summary else {}),
            }
        if text_verbosity:
            request_body["text"] = {"verbosity": text_verbosity}

        url = f"{self._base_url}{CODEX_RESPONSES_PATH}"
        async with httpx.AsyncClient(timeout=self._timeout_s) as http:
            creds = await self._load_valid_credentials(http)
            base_instructions = await aget_codex_instructions(
                http, model=request_body["model"]
            )
            instructions_extra_removed = False
            if extra_instructions:
                request_body["instructions"] = (
                    f"{base_instructions}\n\n{extra_instructions}".strip()
                )
            else:
                request_body["instructions"] = base_instructions

            tool_choice_removed = False
            temperature_removed = False
            max_output_tokens_removed = False
            attempt = 0
            while True:
                try:
                    async with http.stream(
                        "POST",
                        url,
                        headers=self._headers(creds),
                        json=request_body,
                    ) as response:
                        if response.status_code >= 400:
                            try:
                                await response.aread()
                            except Exception:
                                pass

                            err = CodexClient._to_api_error(response)

                            if (
                                not instructions_extra_removed
                                and extra_instructions
                                and err.status_code == 400
                                and "instruction" in str(err).lower()
                            ):
                                request_body["instructions"] = base_instructions
                                instructions_extra_removed = True
                                continue

                            if (
                                not tool_choice_removed
                                and tool_choice is not None
                                and err.status_code == 400
                                and "tool_choice" in str(err).lower()
                            ):
                                request_body.pop("tool_choice", None)
                                tool_choice_removed = True
                                continue

                            if (
                                not temperature_removed
                                and temperature is not None
                                and err.status_code == 400
                                and "temperature" in str(err).lower()
                            ):
                                request_body.pop("temperature", None)
                                temperature_removed = True
                                continue

                            if (
                                not max_output_tokens_removed
                                and max_output_tokens is not None
                                and err.status_code == 400
                                and any(
                                    key in str(err).lower()
                                    for key in ("max_output_tokens", "max_tokens")
                                )
                            ):
                                request_body.pop("max_output_tokens", None)
                                max_output_tokens_removed = True
                                continue

                            if (
                                _is_retryable_status(err.status_code)
                                and attempt < self._max_retries
                            ):
                                await asyncio.sleep(_backoff_s(attempt))
                                attempt += 1
                                continue

                            raise err

                        async for event in aiter_sse_events(response.aiter_lines()):
                            yield event
                        return

                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt < self._max_retries:
                        await asyncio.sleep(_backoff_s(attempt))
                        attempt += 1
                        continue
                    raise CodexAPIError(
                        "Network error calling Codex backend", status_code=None
                    ) from exc

    async def acomplete_with_response(
        self,
        *,
        input_items: list[InputItem],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        text_verbosity: str | None = None,
        include: list[str] | None = None,
        extra_instructions: str | None = None,
    ) -> CompletionResult:
        last_response: object | None = None
        async for event in self.astream_events(
            input_items=input_items,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            text_verbosity=text_verbosity,
            include=include,
            extra_instructions=extra_instructions,
        ):
            if is_terminal_event(event):
                last_response = event.get("response")
                break

        return CompletionResult(
            parsed=parse_assistant_message(last_response),
            response=last_response,
        )

    async def acomplete(
        self,
        *,
        input_items: list[InputItem],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        text_verbosity: str | None = None,
        include: list[str] | None = None,
        extra_instructions: str | None = None,
    ) -> ParsedAssistantMessage:
        return (
            await self.acomplete_with_response(
                input_items=input_items,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                text_verbosity=text_verbosity,
                include=include,
                extra_instructions=extra_instructions,
            )
        ).parsed

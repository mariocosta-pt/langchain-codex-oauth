from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

from codex_oauth.auth import (
    decode_jwt_payload,
    extract_chatgpt_account_id,
    refresh_access_token,
)
from codex_oauth.exceptions import CodexAPIError, NotAuthenticatedError
from codex_oauth.instructions import get_codex_instructions
from codex_oauth.models import ChatMessage, messages_to_input, normalize_model
from codex_oauth.sse import extract_text_delta, is_terminal_event, iter_sse_events
from codex_oauth.store import AuthStore, OAuthCredentials

CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_RESPONSES_PATH = "/codex/responses"

DEFAULT_INCLUDE = ["reasoning.encrypted_content"]


class CodexClient:
    def __init__(
        self,
        auth_store: AuthStore | None = None,
        *,
        base_url: str = CODEX_BASE_URL,
        timeout_s: float = 60.0,
    ) -> None:
        self._store = auth_store or AuthStore()
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

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
        request_body: dict[str, Any] = {
            "model": normalize_model(model),
            "store": False,
            "stream": True,
            "input": messages_to_input(messages),
            "include": include or DEFAULT_INCLUDE,
        }
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

            request_body["instructions"] = get_codex_instructions(
                http, model=request_body["model"]
            )

            with http.stream(
                "POST",
                url,
                headers=self._headers(creds),
                json=request_body,
            ) as response:
                if response.status_code >= 400:
                    # Ensure error body is read so response.text is populated.
                    try:
                        response.read()
                    except Exception:
                        pass
                    raise self._to_api_error(response)

                for event in iter_sse_events(response.iter_lines()):
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
        return "".join(
            self.stream_chat(
                messages=messages,
                model=model,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                text_verbosity=text_verbosity,
                include=include,
            )
        )

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
        # Normalize these to 429 so downstream code can treat them like rate limits.
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

from pathlib import Path

import httpx
import pytest

from codex_oauth.instructions import INSTRUCTIONS_MODE_ENV, get_codex_instructions


def _error_transport(_: httpx.Request) -> httpx.Response:
    return httpx.Response(500, content=b"error")


def test_bundled_instructions_mode_returns_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGCHAIN_CODEX_OAUTH_HOME", str(tmp_path))
    monkeypatch.setenv(INSTRUCTIONS_MODE_ENV, "bundled")

    with httpx.Client(transport=httpx.MockTransport(_error_transport)) as http:
        text = get_codex_instructions(http, model="gpt-5.2-codex")

    assert isinstance(text, str)
    assert len(text) > 100


def test_gpt_56_models_use_bundled_gpt_56_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGCHAIN_CODEX_OAUTH_HOME", str(tmp_path))
    monkeypatch.setenv(INSTRUCTIONS_MODE_ENV, "bundled")

    with httpx.Client(transport=httpx.MockTransport(_error_transport)) as http:
        prompts = [
            get_codex_instructions(http, model=model)
            for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
        ]

    assert all("curious, rich personality" in prompt for prompt in prompts)
    assert len(set(prompts)) == 1


def test_gpt_56_github_mode_extracts_model_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGCHAIN_CODEX_OAUTH_HOME", str(tmp_path))
    monkeypatch.setenv(INSTRUCTIONS_MODE_ENV, "github")

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url == "https://api.github.com/repos/openai/codex/releases/latest":
            return httpx.Response(200, json={"tag_name": "rust-v-test"})
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "gpt-5.6-terra",
                        "model_messages": {
                            "instructions_template": "You are Codex for Terra."
                        },
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        text = get_codex_instructions(http, model="gpt-5.6-terra")

    assert text == "You are Codex for Terra."
    assert (tmp_path / "cache" / "gpt-5.6-terra-instructions.md").exists()


def test_auto_mode_falls_back_to_bundled_when_github_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGCHAIN_CODEX_OAUTH_HOME", str(tmp_path))
    monkeypatch.delenv(INSTRUCTIONS_MODE_ENV, raising=False)

    with httpx.Client(transport=httpx.MockTransport(_error_transport)) as http:
        text = get_codex_instructions(http, model="gpt-5.2-codex")

    assert isinstance(text, str)
    assert len(text) > 100


def test_cache_mode_raises_if_cache_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGCHAIN_CODEX_OAUTH_HOME", str(tmp_path))
    monkeypatch.setenv(INSTRUCTIONS_MODE_ENV, "cache")

    with httpx.Client(transport=httpx.MockTransport(_error_transport)) as http:
        with pytest.raises(RuntimeError):
            get_codex_instructions(http, model="gpt-5.2-codex")

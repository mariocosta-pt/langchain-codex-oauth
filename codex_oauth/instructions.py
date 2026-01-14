from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

import httpx

from codex_oauth.models import normalize_model

_GITHUB_API_RELEASES = "https://api.github.com/repos/openai/codex/releases/latest"
_GITHUB_HTML_RELEASES = "https://github.com/openai/codex/releases/latest"

INSTRUCTIONS_MODE_ENV = "LANGCHAIN_CODEX_OAUTH_INSTRUCTIONS_MODE"
InstructionsMode = Literal["auto", "cache", "github", "bundled"]

# Bundled fallback shipped with the package.
DEFAULT_BUNDLED_CACHE_FILE = "gpt-5.2-codex-instructions.md"


def _home_dir() -> Path:
    env_home = os.environ.get("LANGCHAIN_CODEX_OAUTH_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".langchain-codex-oauth"


def _cache_dir() -> Path:
    return _home_dir() / "cache"


@dataclass(frozen=True)
class PromptFamily:
    family: str
    prompt_file: str
    cache_file: str


_FAMILIES: list[PromptFamily] = [
    PromptFamily(
        family="gpt-5.2-codex",
        prompt_file="gpt-5.2-codex_prompt.md",
        cache_file="gpt-5.2-codex-instructions.md",
    ),
    PromptFamily(
        family="codex-max",
        prompt_file="gpt-5.1-codex-max_prompt.md",
        cache_file="codex-max-instructions.md",
    ),
    PromptFamily(
        family="codex",
        prompt_file="gpt_5_codex_prompt.md",
        cache_file="codex-instructions.md",
    ),
    PromptFamily(
        family="gpt-5.2",
        prompt_file="gpt_5_2_prompt.md",
        cache_file="gpt-5.2-instructions.md",
    ),
    PromptFamily(
        family="gpt-5.1",
        prompt_file="gpt_5_1_prompt.md",
        cache_file="gpt-5.1-instructions.md",
    ),
]


def _model_family(model: str) -> PromptFamily:
    model_id = normalize_model(model).lower()

    if "gpt-5.2-codex" in model_id or "gpt 5.2 codex" in model_id:
        return _FAMILIES[0]
    if "codex-max" in model_id:
        return _FAMILIES[1]
    if "codex" in model_id or model_id.startswith("codex-"):
        return _FAMILIES[2]
    if "gpt-5.2" in model_id:
        return _FAMILIES[3]
    return _FAMILIES[4]


def _instructions_mode() -> InstructionsMode:
    raw = (os.environ.get(INSTRUCTIONS_MODE_ENV) or "auto").strip().lower()
    if raw in {"auto", "cache", "github", "bundled"}:
        return raw  # type: ignore[return-value]
    return "auto"


def _load_bundled(cache_file: str) -> str | None:
    try:
        pkg = resources.files("codex_oauth.bundled_prompts")
        path = pkg / cache_file
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _load_bundled_for_family(family: PromptFamily) -> str:
    text = _load_bundled(family.cache_file)
    if text:
        return text
    fallback = _load_bundled(DEFAULT_BUNDLED_CACHE_FILE)
    if fallback:
        return fallback
    raise RuntimeError(
        "No bundled instructions found. Reinstall the package or set "
        f"`{INSTRUCTIONS_MODE_ENV}=github` to fetch instructions."
    )


def _latest_release_tag(http: httpx.Client) -> str:
    try:
        response = http.get(_GITHUB_API_RELEASES, timeout=15.0)
        if response.status_code == 200:
            data = response.json()
            tag = data.get("tag_name") if isinstance(data, dict) else None
            if isinstance(tag, str) and tag:
                return tag
    except Exception:
        pass

    response = http.get(_GITHUB_HTML_RELEASES, follow_redirects=True, timeout=15.0)
    response.raise_for_status()

    final_url = str(response.url)
    if "/tag/" in final_url:
        tag = final_url.rsplit("/tag/", 1)[-1]
        if tag and "/" not in tag:
            return tag

    text = response.text
    marker = "/openai/codex/releases/tag/"
    idx = text.find(marker)
    if idx >= 0:
        tail = text[idx + len(marker) :]
        tag = tail.split('"', 1)[0]
        if tag:
            return tag

    raise RuntimeError("Failed to determine latest Codex release tag")


async def _alatest_release_tag(http: httpx.AsyncClient) -> str:
    try:
        response = await http.get(_GITHUB_API_RELEASES, timeout=15.0)
        if response.status_code == 200:
            data = response.json()
            tag = data.get("tag_name") if isinstance(data, dict) else None
            if isinstance(tag, str) and tag:
                return tag
    except Exception:
        pass

    response = await http.get(
        _GITHUB_HTML_RELEASES, follow_redirects=True, timeout=15.0
    )
    response.raise_for_status()

    final_url = str(response.url)
    if "/tag/" in final_url:
        tag = final_url.rsplit("/tag/", 1)[-1]
        if tag and "/" not in tag:
            return tag

    text = response.text
    marker = "/openai/codex/releases/tag/"
    idx = text.find(marker)
    if idx >= 0:
        tail = text[idx + len(marker) :]
        tag = tail.split('"', 1)[0]
        if tag:
            return tag

    raise RuntimeError("Failed to determine latest Codex release tag")


def _write_cache(
    cache_path: Path,
    meta_path: Path,
    *,
    tag: str,
    url: str,
    etag: str | None,
    text: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "etag": etag,
                "tag": tag,
                "last_checked_ms": int(time.time() * 1000),
                "url": url,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def get_codex_instructions(http: httpx.Client, *, model: str) -> str:
    """Resolve the `instructions` payload for the Codex backend.

    Default behavior is offline-friendly: if a cached prompt exists, use it.
    If not, try GitHub; if GitHub fails, fall back to bundled prompts.
    """

    mode = _instructions_mode()
    family = _model_family(model)

    cache_dir = _cache_dir()
    cache_path = cache_dir / family.cache_file
    meta_path = cache_dir / (family.cache_file.replace(".md", "-meta.json"))

    if mode in {"auto", "cache"} and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    if mode == "cache":
        raise RuntimeError(
            f"Instructions cache is missing ({cache_path}). "
            f"Set `{INSTRUCTIONS_MODE_ENV}=github` to fetch, or "
            f"`{INSTRUCTIONS_MODE_ENV}=bundled`."
        )

    if mode == "bundled":
        return _load_bundled_for_family(family)

    # mode == github or auto with no cache
    try:
        tag = _latest_release_tag(http)
        url = f"https://raw.githubusercontent.com/openai/codex/{tag}/codex-rs/core/{family.prompt_file}"
        response = http.get(url, timeout=30.0)
        response.raise_for_status()
        _write_cache(
            cache_path,
            meta_path,
            tag=tag,
            url=url,
            etag=response.headers.get("etag"),
            text=response.text,
        )
        return response.text
    except Exception:
        if mode == "github":
            raise
        return _load_bundled_for_family(family)


async def aget_codex_instructions(http: httpx.AsyncClient, *, model: str) -> str:
    """Async variant of `get_codex_instructions`."""

    mode = _instructions_mode()
    family = _model_family(model)

    cache_dir = _cache_dir()
    cache_path = cache_dir / family.cache_file
    meta_path = cache_dir / (family.cache_file.replace(".md", "-meta.json"))

    if mode in {"auto", "cache"} and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    if mode == "cache":
        raise RuntimeError(
            f"Instructions cache is missing ({cache_path}). "
            f"Set `{INSTRUCTIONS_MODE_ENV}=github` to fetch, or "
            f"`{INSTRUCTIONS_MODE_ENV}=bundled`."
        )

    if mode == "bundled":
        return _load_bundled_for_family(family)

    try:
        tag = await _alatest_release_tag(http)
        url = f"https://raw.githubusercontent.com/openai/codex/{tag}/codex-rs/core/{family.prompt_file}"
        response = await http.get(url, timeout=30.0)
        response.raise_for_status()
        _write_cache(
            cache_path,
            meta_path,
            tag=tag,
            url=url,
            etag=response.headers.get("etag"),
            text=response.text,
        )
        return response.text
    except Exception:
        if mode == "github":
            raise
        return _load_bundled_for_family(family)

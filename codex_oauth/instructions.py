from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from codex_oauth.models import normalize_model

_GITHUB_API_RELEASES = "https://api.github.com/repos/openai/codex/releases/latest"
_GITHUB_HTML_RELEASES = "https://github.com/openai/codex/releases/latest"


class _CacheMeta(dict):
    etag: str | None
    tag: str
    last_checked_ms: int
    url: str


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

    # If redirected to /tag/<tag>, use that.
    final_url = str(response.url)
    if "/tag/" in final_url:
        tag = final_url.rsplit("/tag/", 1)[-1]
        if tag and "/" not in tag:
            return tag

    # Fallback: try regex-less parsing.
    text = response.text
    marker = "/openai/codex/releases/tag/"
    idx = text.find(marker)
    if idx >= 0:
        tail = text[idx + len(marker) :]
        tag = tail.split('"', 1)[0]
        if tag:
            return tag

    raise RuntimeError("Failed to determine latest Codex release tag")


def get_codex_instructions(http: httpx.Client, *, model: str) -> str:
    """Fetch and cache Codex CLI model-family instructions.

    The ChatGPT/Codex backend validates `instructions` and expects Codex CLI style
    prompts. We fetch the latest prompt from `openai/codex` releases, cache it on
    disk, and reuse it across requests.
    """

    family = _model_family(model)
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / family.cache_file
    meta_path = cache_dir / (family.cache_file.replace(".md", "-meta.json"))

    cached_etag: str | None = None
    cached_tag: str | None = None
    cached_checked_ms: int | None = None

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                cached_etag = (
                    meta.get("etag") if isinstance(meta.get("etag"), str) else None
                )
                cached_tag = (
                    meta.get("tag") if isinstance(meta.get("tag"), str) else None
                )
                cached_checked_ms = (
                    int(meta.get("last_checked_ms"))
                    if isinstance(meta.get("last_checked_ms"), int)
                    else None
                )
        except Exception:
            cached_etag = None
            cached_tag = None
            cached_checked_ms = None

    # Rate limit protection: if checked recently and cache exists, use it.
    ttl_ms = 15 * 60 * 1000
    now_ms = int(time.time() * 1000)
    if (
        cached_checked_ms
        and cache_path.exists()
        and (now_ms - cached_checked_ms) < ttl_ms
    ):
        return cache_path.read_text(encoding="utf-8")

    tag = _latest_release_tag(http)
    if cached_tag != tag:
        cached_etag = None

    url = f"https://raw.githubusercontent.com/openai/codex/{tag}/codex-rs/core/{family.prompt_file}"

    headers: dict[str, str] = {}
    if cached_etag:
        headers["If-None-Match"] = cached_etag

    response = http.get(url, headers=headers, timeout=30.0)

    if response.status_code == 304 and cache_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "etag": cached_etag,
                    "tag": tag,
                    "last_checked_ms": now_ms,
                    "url": url,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return cache_path.read_text(encoding="utf-8")

    response.raise_for_status()

    instructions = response.text
    etag = response.headers.get("etag")

    cache_path.write_text(instructions, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "etag": etag,
                "tag": tag,
                "last_checked_ms": now_ms,
                "url": url,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return instructions

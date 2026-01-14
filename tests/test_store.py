from pathlib import Path

import pytest

from codex_oauth.store import AuthStore, OAuthCredentials


def test_store_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGCHAIN_CODEX_OAUTH_AUTH_PATH", str(tmp_path / "auth.json"))
    store = AuthStore()
    creds = OAuthCredentials(
        access="a",
        refresh="r",
        expires=123,
        account_id="acct",
    )
    store.save(creds)
    loaded = store.load()
    assert loaded == creds

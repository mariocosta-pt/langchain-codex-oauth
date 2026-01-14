from codex_oauth.models import normalize_model


def test_normalize_model_strips_provider_prefix() -> None:
    assert normalize_model("openai/gpt-5.2-codex") == "gpt-5.2-codex"
    assert normalize_model("gpt-5.2-codex") == "gpt-5.2-codex"

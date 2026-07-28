from langchain_codex_oauth.chat_models import ChatCodexOAuth


def test_gpt_56_models_are_preserved() -> None:
    for model_id in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        assert ChatCodexOAuth(model=model_id).model == model_id


def test_default_model_is_gpt_56_sol() -> None:
    assert ChatCodexOAuth().model == "gpt-5.6-sol"


def test_provider_prefix_is_removed_from_model() -> None:
    model = ChatCodexOAuth(model="openai-codex/gpt-5.6-terra")

    assert model.model == "gpt-5.6-terra"


def test_gpt_56_reasoning_effort_suffixes() -> None:
    for effort in ("low", "medium", "high", "xhigh", "max"):
        model = ChatCodexOAuth(model=f"gpt-5.6-sol-{effort}")

        assert model.model == "gpt-5.6-sol"
        assert model.reasoning_effort == effort


def test_legacy_codex_max_model_is_not_parsed_as_reasoning_suffix() -> None:
    model = ChatCodexOAuth(model="gpt-5.1-codex-max")

    assert model.model == "gpt-5.1-codex-max"
    assert model.reasoning_effort == "medium"


def test_reasoning_dict_preserves_suffix_effort_when_effort_omitted() -> None:
    model = ChatCodexOAuth(
        model="gpt-5.5-low",
        reasoning={"summary": "auto"},
        verbosity="medium",
    )

    assert model.model == "gpt-5.5"
    assert model.reasoning_effort == "low"
    assert model.reasoning_summary == "auto"
    assert model.text_verbosity == "medium"


def test_reasoning_dict_can_override_legacy_reasoning_fields() -> None:
    model = ChatCodexOAuth(
        model="gpt-5.5-low",
        reasoning={"effort": "high", "summary": "detailed"},
        reasoning_effort="medium",
        reasoning_summary="concise",
    )

    assert model.model == "gpt-5.5"
    assert model.reasoning_effort == "high"
    assert model.reasoning_summary == "detailed"


def test_verbosity_alias_sets_text_verbosity() -> None:
    model = ChatCodexOAuth(model="gpt-5.5", verbosity="high")

    assert model.text_verbosity == "high"

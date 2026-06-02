from langchain_codex_oauth.chat_models import ChatCodexOAuth


def test_model_suffix_sets_reasoning_effort_and_base_model() -> None:
    model = ChatCodexOAuth(model="gpt-5.5-xhigh")

    assert model.model == "gpt-5.5"
    assert model.reasoning_effort == "xhigh"


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

from codex_oauth.models import message_item


def test_message_item_user_uses_input_text() -> None:
    item = message_item("user", "hi")
    assert item["content"][0]["type"] == "input_text"


def test_message_item_assistant_uses_output_text() -> None:
    item = message_item("assistant", "hi")
    assert item["content"][0]["type"] == "output_text"

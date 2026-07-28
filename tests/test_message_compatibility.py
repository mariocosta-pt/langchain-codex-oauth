from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from codex_oauth.models import function_call_output_item
from langchain_codex_oauth.chat_models import _to_input_items


def test_to_input_items_extracts_human_text_content_blocks() -> None:
    items = _to_input_items(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "input_text", "text": "world"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ]
            )
        ]
    )

    assert items == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello\nworld"}],
        }
    ]


def test_to_input_items_extracts_system_text_content_blocks() -> None:
    items = _to_input_items(
        [SystemMessage(content=[{"type": "text", "text": "be concise"}])],
        system_prompt_mode="default",
    )

    assert items == [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "be concise"}],
        }
    ]


def test_to_input_items_preserves_assistant_text_and_tool_calls() -> None:
    tool_call = {"name": "search", "args": {"query": "Gatsby"}, "id": "call_1"}

    items = _to_input_items(
        [
            AIMessage(
                content=[{"type": "text", "text": "I need to search."}],
                tool_calls=[tool_call],
            )
        ]
    )

    assert items[0] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "I need to search."}],
    }
    assert items[1] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "search",
        "arguments": '{"query":"Gatsby"}',
    }


def test_tool_message_non_string_outputs_are_json_serialized() -> None:
    assert function_call_output_item("call_1", {"ok": True, "count": 2}) == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"ok": true, "count": 2}',
    }


def test_to_input_items_preserves_tool_message_string_output() -> None:
    message = ToolMessage(content="plain output", tool_call_id="call_1")
    items = _to_input_items([message])

    assert items == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "plain output",
        }
    ]

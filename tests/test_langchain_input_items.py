from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_codex_oauth.chat_models import _to_input_items


def test_to_input_items_includes_tool_loop_items() -> None:
    tool_call = {"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1"}

    messages = [
        HumanMessage(content="use the tool"),
        AIMessage(content="", tool_calls=[tool_call]),
        ToolMessage(content="3", tool_call_id="call_1"),
    ]

    items = _to_input_items(messages)

    assert items[0]["type"] == "message"
    assert items[0]["role"] == "user"

    # function_call item emitted for AI tool call
    assert any(
        item.get("type") == "function_call" and item.get("call_id") == "call_1"
        for item in items
    )

    # function_call_output item emitted for ToolMessage
    assert any(
        item.get("type") == "function_call_output" and item.get("call_id") == "call_1"
        for item in items
    )

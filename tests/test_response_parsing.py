from codex_oauth.response import parse_assistant_message


def test_parse_assistant_message_with_tool_call() -> None:
    response = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello."}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"location":"Paris"}',
            },
        ]
    }

    parsed = parse_assistant_message(response)
    assert parsed.content == "Hello."
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["id"] == "call_1"
    assert parsed.tool_calls[0]["name"] == "get_weather"
    assert parsed.tool_calls[0]["args"] == {"location": "Paris"}


def test_parse_assistant_message_invalid_tool_args() -> None:
    response = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "calculator",
                "arguments": "{not json}",
            }
        ]
    }

    parsed = parse_assistant_message(response)
    assert parsed.tool_calls == []
    assert len(parsed.invalid_tool_calls) == 1
    assert parsed.invalid_tool_calls[0]["id"] == "call_2"
    assert parsed.invalid_tool_calls[0]["name"] == "calculator"

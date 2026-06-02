from codex_oauth.client import _accumulate_response_event, _response_from_stream_events
from codex_oauth.response import parse_assistant_message


def test_accumulates_function_call_when_terminal_response_omits_output() -> None:
    output_items: dict[int, dict] = {}
    text_parts: list[str] = []

    _accumulate_response_event(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "fetch_text_from_url",
            },
        },
        output_items,
        text_parts,
    )
    _accumulate_response_event(
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "call_id": "call_1",
            "delta": '{"url":"https://example.com',
        },
        output_items,
        text_parts,
    )
    _accumulate_response_event(
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "call_id": "call_1",
            "delta": '/file.txt"}',
        },
        output_items,
        text_parts,
    )

    response = _response_from_stream_events(
        {"id": "resp_1", "status": "completed"}, output_items, text_parts
    )

    parsed = parse_assistant_message(response)
    assert parsed.content == ""
    assert parsed.invalid_tool_calls == []
    assert parsed.tool_calls == [
        {
            "type": "tool_call",
            "id": "call_1",
            "name": "fetch_text_from_url",
            "args": {"url": "https://example.com/file.txt"},
        }
    ]


def test_terminal_response_output_takes_precedence() -> None:
    output_items = {
        0: {
            "type": "function_call",
            "call_id": "call_stream",
            "name": "stream_tool",
            "arguments": "{}",
        }
    }

    response = _response_from_stream_events(
        {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ]
        },
        output_items,
        [],
    )

    parsed = parse_assistant_message(response)
    assert parsed.content == "done"
    assert parsed.tool_calls == []


def test_accumulates_text_delta_when_terminal_response_omits_output() -> None:
    output_items: dict[int, dict] = {}
    text_parts: list[str] = []

    _accumulate_response_event(
        {"type": "response.output_text.delta", "delta": "hel"},
        output_items,
        text_parts,
    )
    _accumulate_response_event(
        {"type": "response.output_text.delta", "delta": "lo"},
        output_items,
        text_parts,
    )

    response = _response_from_stream_events(None, output_items, text_parts)

    parsed = parse_assistant_message(response)
    assert parsed.content == "hello"
    assert parsed.tool_calls == []

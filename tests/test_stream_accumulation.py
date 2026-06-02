from codex_oauth.client import (
    _StreamResponseState,
    _accumulate_response_event,
    _response_from_stream_events,
)
from codex_oauth.response import parse_assistant_message


def test_accumulates_function_call_when_terminal_response_omits_output() -> None:
    stream_state = _StreamResponseState()

    _accumulate_response_event(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "fetch_text_from_url",
            },
        },
        stream_state,
    )
    _accumulate_response_event(
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc_1",
            "delta": '{"url":"https://example.com',
        },
        stream_state,
    )
    _accumulate_response_event(
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc_1",
            "delta": '/file.txt"}',
        },
        stream_state,
    )

    response = _response_from_stream_events(
        {"id": "resp_1", "status": "completed"}, stream_state
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


def test_accumulates_arguments_by_item_id_when_output_index_changes() -> None:
    stream_state = _StreamResponseState()

    _accumulate_response_event(
        {
            "type": "response.output_item.added",
            "output_index": 3,
            "item": {
                "type": "function_call",
                "id": "fc_weather",
                "call_id": "call_weather",
                "name": "get_weather",
            },
        },
        stream_state,
    )
    _accumulate_response_event(
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc_weather",
            "delta": '{"city":"Lis',
        },
        stream_state,
    )
    _accumulate_response_event(
        {
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": "fc_weather",
            "arguments": '{"city":"Lisbon"}',
        },
        stream_state,
    )

    response = _response_from_stream_events(None, stream_state)

    parsed = parse_assistant_message(response)
    assert parsed.tool_calls == [
        {
            "type": "tool_call",
            "id": "call_weather",
            "name": "get_weather",
            "args": {"city": "Lisbon"},
        }
    ]


def test_accumulates_arguments_before_full_output_item_metadata() -> None:
    stream_state = _StreamResponseState()

    _accumulate_response_event(
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc_early",
            "delta": '{"query":"Gat',
        },
        stream_state,
    )
    _accumulate_response_event(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_early",
                "call_id": "call_early",
                "name": "search",
            },
        },
        stream_state,
    )
    _accumulate_response_event(
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc_early",
            "delta": 'sby"}',
        },
        stream_state,
    )

    response = _response_from_stream_events(None, stream_state)

    parsed = parse_assistant_message(response)
    assert parsed.tool_calls == [
        {
            "type": "tool_call",
            "id": "call_early",
            "name": "search",
            "args": {"query": "Gatsby"},
        }
    ]


def test_accumulates_parallel_tool_calls_in_first_seen_order() -> None:
    stream_state = _StreamResponseState()

    _accumulate_response_event(
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": "fc_second",
                "call_id": "call_second",
                "name": "second_tool",
            },
        },
        stream_state,
    )
    _accumulate_response_event(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_first",
                "call_id": "call_first",
                "name": "first_tool",
            },
        },
        stream_state,
    )
    _accumulate_response_event(
        {
            "type": "response.function_call_arguments.done",
            "output_index": 1,
            "item_id": "fc_second",
            "arguments": '{"value":2}',
        },
        stream_state,
    )
    _accumulate_response_event(
        {
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": "fc_first",
            "arguments": '{"value":1}',
        },
        stream_state,
    )

    response = _response_from_stream_events(None, stream_state)

    parsed = parse_assistant_message(response)
    assert [call["name"] for call in parsed.tool_calls] == [
        "second_tool",
        "first_tool",
    ]
    assert [call["args"] for call in parsed.tool_calls] == [
        {"value": 2},
        {"value": 1},
    ]


def test_terminal_response_output_takes_precedence() -> None:
    stream_state = _StreamResponseState()
    _accumulate_response_event(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_stream",
                "name": "stream_tool",
                "arguments": "{}",
            },
        },
        stream_state,
    )

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
        stream_state,
    )

    parsed = parse_assistant_message(response)
    assert parsed.content == "done"
    assert parsed.tool_calls == []


def test_accumulates_text_delta_when_terminal_response_omits_output() -> None:
    stream_state = _StreamResponseState()

    _accumulate_response_event(
        {"type": "response.output_text.delta", "delta": "hel"}, stream_state
    )
    _accumulate_response_event(
        {"type": "response.output_text.delta", "delta": "lo"}, stream_state
    )

    response = _response_from_stream_events(None, stream_state)

    parsed = parse_assistant_message(response)
    assert parsed.content == "hello"
    assert parsed.tool_calls == []

from codex_oauth.response import extract_response_metadata, parse_assistant_message


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


def test_extract_response_metadata_preserves_reasoning_summary() -> None:
    response = {
        "id": "resp_1",
        "model": "gpt-5.5",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "secret-ciphertext",
                "summary": [
                    {"type": "summary_text", "text": "Checked the tool result."},
                    {"type": "summary_text", "text": "Computed the final answer."},
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        ],
    }

    metadata = extract_response_metadata(response)

    assert metadata["reasoning"]["summary"] == (
        "Checked the tool result.\n\nComputed the final answer."
    )
    assert metadata["reasoning"]["items"][0]["id"] == "rs_1"
    assert metadata["reasoning"]["items"][0]["encrypted_content_present"] is True
    assert "secret-ciphertext" not in str(metadata)


def test_extract_response_metadata_deduplicates_reasoning_summary() -> None:
    response = {
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Used grep results."}],
            },
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Used grep results."}],
            },
        ]
    }

    metadata = extract_response_metadata(response)

    assert len(metadata["reasoning"]["items"]) == 1
    assert metadata["reasoning"]["summary"] == "Used grep results."

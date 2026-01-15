from codex_oauth.sse import extract_text_delta, is_terminal_event, iter_sse_events


def test_iter_sse_events_parses_data_lines() -> None:
    lines = [
        'data: {"type": "response.output_text.delta", "delta": "hi"}\n',
        "\n",
        'data: {"type": "response.done", "response": {}}\n',
        "\n",
    ]
    events = list(iter_sse_events(lines))
    assert [e["type"] for e in events] == [
        "response.output_text.delta",
        "response.done",
    ]


def test_text_delta_extraction() -> None:
    event = {"type": "response.output_text.delta", "delta": "hello"}
    assert extract_text_delta(event) == "hello"


def test_text_delta_ignores_reasoning_like_deltas() -> None:
    event = {"type": "response.reasoning_summary.delta", "delta": "secret"}
    assert extract_text_delta(event) is None


def test_terminal_event_detection() -> None:
    assert is_terminal_event({"type": "response.done"})
    assert is_terminal_event({"type": "response.completed"})
    assert not is_terminal_event({"type": "response.output_text.delta"})

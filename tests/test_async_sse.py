import asyncio

from codex_oauth.sse import aiter_sse_events


def test_aiter_sse_events_parses_data_lines() -> None:
    lines = [
        'data: {"type": "response.output_text.delta", "delta": "hi"}\n',
        "\n",
        'data: {"type": "response.done", "response": {}}\n',
        "\n",
    ]

    async def _run() -> list[dict]:
        async def _gen():
            for line in lines:
                yield line

        events: list[dict] = []
        async for event in aiter_sse_events(_gen()):
            events.append(event)
        return events

    events = asyncio.run(_run())
    assert [e["type"] for e in events] == [
        "response.output_text.delta",
        "response.done",
    ]

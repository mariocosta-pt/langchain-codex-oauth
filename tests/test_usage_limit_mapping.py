import httpx

from codex_oauth.client import CodexClient


def test_usage_limit_404_is_mapped_to_429() -> None:
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    response = httpx.Response(
        404,
        request=request,
        content=b'{"error": {"code": "usage_limit_reached"}}',
    )

    err = CodexClient._to_api_error(response)
    assert err.status_code == 429
    assert "usage limit" in str(err).lower()

import base64
import json

from codex_oauth.auth import decode_jwt_payload, extract_chatgpt_account_id


def _b64url(obj: dict) -> str:
    raw = json.dumps(obj).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def test_decode_jwt_payload_and_extract_account_id() -> None:
    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"}}
    token = f"aaa.{_b64url(payload)}.bbb"
    decoded = decode_jwt_payload(token)
    assert decoded
    assert extract_chatgpt_account_id(decoded) == "acct_123"

from __future__ import annotations

import argparse
import datetime as dt

from codex_oauth.auth import (
    decode_jwt_payload,
    exchange_authorization_code,
    extract_chatgpt_account_id,
    login_manual,
    login_via_browser,
)
from codex_oauth.exceptions import OAuthFlowError
from codex_oauth.store import AuthStore, OAuthCredentials
from langchain_codex_oauth.version import __version__


def _format_ms(ms: int) -> str:
    if not ms:
        return "unknown"
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).isoformat()


def _cmd_auth_login(args: argparse.Namespace) -> int:
    if args.manual:
        code, verifier = login_manual()
    else:
        result = login_via_browser(timeout_s=args.timeout_s)
        if not result:
            raise OAuthFlowError(
                "OAuth callback timed out. Re-run with --manual, or try again."
            )
        code, verifier = result

    token = exchange_authorization_code(code=code, verifier=verifier)
    payload = decode_jwt_payload(token.access)
    if not payload:
        raise OAuthFlowError("Received invalid access token")
    account_id = extract_chatgpt_account_id(payload)
    if not account_id:
        raise OAuthFlowError("Failed to extract chatgpt_account_id from token")

    store = AuthStore()
    store.save(
        OAuthCredentials(
            access=token.access,
            refresh=token.refresh,
            expires=token.expires_at_ms,
            account_id=account_id,
        )
    )
    print("Login successful. Credentials saved.")
    return 0


def _cmd_auth_status(_: argparse.Namespace) -> int:
    store = AuthStore()
    try:
        creds = store.load()
    except Exception as exc:
        print(str(exc))
        return 1

    expires = _format_ms(creds.expires)
    print("Logged in: yes")
    print(f"Account id: {creds.account_id}")
    print(f"Expires (UTC): {expires}")
    return 0


def _cmd_auth_logout(_: argparse.Namespace) -> int:
    store = AuthStore()
    store.delete()
    print("Logged out.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="langchain-codex-oauth")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    auth = sub.add_parser("auth", help="Manage OAuth credentials")
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)

    login = auth_sub.add_parser("login", help="Login via ChatGPT OAuth")
    login.add_argument("--manual", action="store_true", help="Paste redirect URL")
    login.add_argument("--timeout-s", type=int, default=180)
    login.set_defaults(func=_cmd_auth_login)

    status = auth_sub.add_parser("status", help="Show auth status")
    status.set_defaults(func=_cmd_auth_status)

    logout = auth_sub.add_parser("logout", help="Remove local credentials")
    logout.set_defaults(func=_cmd_auth_logout)

    args = parser.parse_args()
    try:
        rc = int(args.func(args))
    except OAuthFlowError as exc:
        print(str(exc))
        rc = 2
    raise SystemExit(rc)

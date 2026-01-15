from __future__ import annotations

import os


def get_env_str(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_env_int(name: str) -> int | None:
    value = get_env_str(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def get_env_float(name: str) -> float | None:
    value = get_env_str(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None

from langchain_core.tools import tool

from langchain_codex_oauth.tooling import convert_tools, normalize_tool_choice


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""

    return a + b


def test_convert_tools_function_schema() -> None:
    schemas = convert_tools([add])
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["type"] == "function"
    assert schema["name"] == "add"
    assert "parameters" in schema


def test_normalize_tool_choice() -> None:
    assert normalize_tool_choice(None) is None
    assert normalize_tool_choice("any") == "auto"
    assert normalize_tool_choice("auto") == "auto"
    assert normalize_tool_choice("required") == "required"

    forced = normalize_tool_choice("add")
    assert forced["type"] == "function"
    assert forced["name"] == "add"

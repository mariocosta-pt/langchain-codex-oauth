from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

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
    assert normalize_tool_choice("any") == "required"
    assert normalize_tool_choice("auto") == "auto"
    assert normalize_tool_choice("required") == "required"

    forced = normalize_tool_choice("add")
    assert forced["type"] == "function"
    assert forced["name"] == "add"


def test_convert_tools_accepts_openai_tool_schema_dict() -> None:
    class Answer(BaseModel):
        answer: str

    openai_tool_schema = convert_to_openai_tool(Answer)
    schemas = convert_tools([openai_tool_schema])

    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["name"] == "Answer"
    assert "parameters" in schemas[0]

import pytest

from learning_agent.tool_registry import ToolRegistry
from learning_agent.tools import CalculatorTool, WordCountTool


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


def test_register_and_get_tool(registry: ToolRegistry) -> None:
    calculator = CalculatorTool()

    registry.register(calculator)

    assert registry.get("calculator") is calculator


def test_get_unknown_tool_raises_error(
    registry: ToolRegistry,
) -> None:
    with pytest.raises(
        ValueError,
        match="Onbekende tool: unknown",
    ):
        registry.get("unknown")


def test_register_duplicate_tool_raises_error(
    registry: ToolRegistry,
) -> None:
    registry.register(CalculatorTool())

    with pytest.raises(
        ValueError,
        match="Tool is al geregistreerd: calculator",
    ):
        registry.register(CalculatorTool())


def test_definitions_contains_registered_tools(
    registry: ToolRegistry,
) -> None:
    registry.register(CalculatorTool())
    registry.register(WordCountTool())

    definitions = registry.definitions

    assert len(definitions) == 2
    assert {definition["name"] for definition in definitions} == {
        "calculator",
        "count_words",
    }


def test_definitions_is_empty_when_no_tools_are_registered(
    registry: ToolRegistry,
) -> None:
    assert registry.definitions == []

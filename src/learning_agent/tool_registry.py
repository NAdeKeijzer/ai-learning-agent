from typing import Any, Protocol


class Tool(Protocol):
    name: str

    @property
    def definition(self) -> dict[str, Any]: ...

    def execute(self, arguments: dict[str, Any]) -> Any: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool is already registered: {tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ValueError(f"Unknown tool: {name}") from error

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [tool.definition for tool in self._tools.values()]

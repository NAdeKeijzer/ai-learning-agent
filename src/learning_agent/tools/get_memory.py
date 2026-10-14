import logging
from typing import Any

from learning_agent.memory import SessionMemory


logger = logging.getLogger("GetMemory")


class GetMemoryTool:
    def __init__(self, memory: SessionMemory) -> None:
        self.memory = memory

    @property
    def name(self) -> str:
        return "get_memory"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Retrieves a value from session memory by key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key of the memory to retrieve.",
                    }
                },
                "required": ["key"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        key = arguments["key"]
        value = self.memory.get(key)

        if value is None:
            logger.info("No value found in session memory for key: %s", key)
            return f"No memory found for key '{key}'."

        logger.info("Retrieved value from session memory for key: %s", key)
        return value

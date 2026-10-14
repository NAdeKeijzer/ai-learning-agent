import logging
from typing import Any

from learning_agent.memory import SessionMemory


logger = logging.getLogger("Remember")


class RememberTool:
    def __init__(self, memory: SessionMemory) -> None:
        self.memory = memory

    @property
    def name(self) -> str:
        return "remember"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Stores a value in session memory under a key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key used to identify the memory.",
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to remember.",
                    },
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        key = arguments["key"]
        value = arguments["value"]

        self.memory.remember(key, value)
        logger.info("Stored value in session memory under key: %s", key)

        return f"Remembered '{key}'."

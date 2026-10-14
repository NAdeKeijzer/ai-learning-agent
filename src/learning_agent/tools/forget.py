import logging
from typing import Any

from learning_agent.memory import SessionMemory


logger = logging.getLogger("Forget")


class ForgetTool:
    def __init__(self, memory: SessionMemory) -> None:
        self.memory = memory

    @property
    def name(self) -> str:
        return "forget"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Removes a value from session memory by key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key of the memory to remove.",
                    }
                },
                "required": ["key"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        key = arguments["key"]

        self.memory.forget(key)
        logger.info("Removed value from session memory for key: %s", key)

        return f"Forgot '{key}'."

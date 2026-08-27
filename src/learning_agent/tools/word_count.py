import logging
from typing import Any


logger = logging.getLogger("WordCount")


class WordCountTool:
    @property
    def name(self) -> str:
        return "count_words"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Counts the number of words in a text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text whose words should be counted.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        text = arguments["text"]
        word_count = len(text.split())

        logger.info("Word count: %s", word_count)

        return str(word_count)

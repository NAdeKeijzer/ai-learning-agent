import logging
from typing import Any


logger = logging.getLogger("Calculator")


class CalculatorTool:
    name = "calculate"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": (
                "Voer een eenvoudige rekenkundige bewerking uit op twee getallen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "Het eerste getal.",
                    },
                    "b": {
                        "type": "number",
                        "description": "Het tweede getal.",
                    },
                    "operation": {
                        "type": "string",
                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide",
                        ],
                        "description": "De uit te voeren bewerking.",
                    },
                },
                "required": ["a", "b", "operation"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def execute(self, arguments: dict[str, Any]) -> float:
        a = arguments["a"]
        b = arguments["b"]
        operation = arguments["operation"]

        logger.info(
            "Berekening uitvoeren: a=%s, b=%s, operation=%s",
            a,
            b,
            operation,
        )

        match operation:
            case "add":
                result = a + b
            case "subtract":
                result = a - b
            case "multiply":
                result = a * b
            case "divide":
                if b == 0:
                    raise ValueError("Delen door nul is niet toegestaan.")
                result = a / b
            case _:
                raise ValueError(f"Onbekende bewerking: {operation}")

        logger.info("Berekening afgerond: resultaat=%s", result)

        return result
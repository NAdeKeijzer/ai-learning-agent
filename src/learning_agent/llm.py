import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from learning_agent.tools import calculate


class LLMClient:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")

        api_key = os.environ["OPENAI_API_KEY"]
        self.model = os.environ["OPENAI_MODEL"]

        self.client = OpenAI(api_key=api_key)
        self.previous_response_id: str | None = None

        self.instructions = (
            "Je bent een behulpzame docent die AI-concepten helder en "
            "beknopt in het Nederlands uitlegt. "
            "Gebruik de calculator voor rekenkundige bewerkingen."
        )

        self.tools = [
            {
                "type": "function",
                "name": "calculate",
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
        ]

    def ask(self, question: str) -> str:
        response = self._create_response(question)

        tool_outputs: list[dict[str, str]] = []

        for item in response.output:
            if item.type != "function_call":
                continue

            if item.name != "calculate":
                raise ValueError(f"Onbekende tool: {item.name}")

            arguments = json.loads(item.arguments)

            result = calculate(
                a=arguments["a"],
                b=arguments["b"],
                operation=arguments["operation"],
            )

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": str(result),
                }
            )

        if tool_outputs:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                tools=self.tools,
                previous_response_id=response.id,
                input=tool_outputs,
            )

        self.previous_response_id = response.id

        return response.output_text

    def _create_response(self, question: str):
        if self.previous_response_id is None:
            return self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                tools=self.tools,
                input=question,
            )

        return self.client.responses.create(
            model=self.model,
            instructions=self.instructions,
            tools=self.tools,
            input=question,
            previous_response_id=self.previous_response_id,
        )

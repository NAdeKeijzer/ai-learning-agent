import json
import logging

from openai.types.responses import ResponseFunctionToolCall

from learning_agent.llm import LLMClient
from learning_agent.tool_registry import ToolRegistry


logger = logging.getLogger("Agent")


class Agent:
    MAX_STEPS = 5

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.previous_response_id: str | None = None

        self.instructions = (
            "Je bent een behulpzame docent die AI-concepten helder en "
            "beknopt in het Nederlands uitlegt. "
            "Gebruik beschikbare tools wanneer die geschikt zijn voor de vraag. "
            "Gebruik de calculator voor rekenkundige bewerkingen en de "
            "woordenteller voor het tellen van woorden."
        )

        self.tools = self.tool_registry.definitions

    def ask(self, question: str) -> str:
        logger.info("Nieuwe gebruikersvraag ontvangen")

        response = self.llm_client.create_response(
            input_data=question,
            instructions=self.instructions,
            tools=self.tools,
            previous_response_id=self.previous_response_id,
        )

        for step in range(1, self.MAX_STEPS + 1):
            logger.info("Agentstap %s van %s", step, self.MAX_STEPS)

            tool_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            if not tool_calls:
                logger.info("Geen toolaanroepen; eindantwoord ontvangen")
                self.previous_response_id = response.id
                return response.output_text

            logger.info("%s toolaanroep(en) ontvangen", len(tool_calls))

            tool_outputs = [
                self._execute_tool_call(tool_call) for tool_call in tool_calls
            ]

            response = self.llm_client.create_response(
                input_data=tool_outputs,
                instructions=self.instructions,
                tools=self.tools,
                previous_response_id=response.id,
            )

        raise RuntimeError(
            f"Agent heeft het maximum van {self.MAX_STEPS} stappen bereikt."
        )

    def _execute_tool_call(
        self,
        tool_call: ResponseFunctionToolCall,
    ) -> dict[str, str]:
        logger.info("Tool aangevraagd: %s", tool_call.name)

        arguments = json.loads(tool_call.arguments)
        tool = self.tool_registry.get(tool_call.name)
        result = tool.execute(arguments)

        return {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result),
        }

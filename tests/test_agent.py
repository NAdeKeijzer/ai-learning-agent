from dataclasses import dataclass
from typing import Any

import pytest

from learning_agent.agent import Agent
from learning_agent.tool_registry import ToolRegistry
from learning_agent.tools import CalculatorTool


@dataclass
class FakeResponse:
    id: str
    output_text: str
    output: list[Any]


@dataclass
class FakeFunctionCall:
    type: str
    name: str
    arguments: str
    call_id: str


class FakeLLMClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create_response(
        self,
        *,
        input_data: str | list[dict[str, str]],
        instructions: str,
        tools: list[dict[str, Any]],
        previous_response_id: str | None = None,
    ) -> FakeResponse:
        self.calls.append(
            {
                "input_data": input_data,
                "instructions": instructions,
                "tools": tools,
                "previous_response_id": previous_response_id,
            }
        )

        return self.responses.pop(0)


def test_agent_returns_direct_llm_answer() -> None:
    llm_client = FakeLLMClient(
        responses=[
            FakeResponse(
                id="response-1",
                output_text="Kafka is an event streaming platform.",
                output=[],
            )
        ]
    )

    agent = Agent(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
    )

    result = agent.ask("What is Kafka?")

    assert result == "Kafka is an event streaming platform."
    assert agent.previous_response_id == "response-1"
    assert len(llm_client.calls) == 1


def test_agent_executes_tool_call_and_returns_final_answer() -> None:
    llm_client = FakeLLMClient(
        responses=[
            FakeResponse(
                id="response-1",
                output_text="",
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="calculator",
                        arguments='{"expression": "6 * 7"}',
                        call_id="call-1",
                    )
                ],
            ),
            FakeResponse(
                id="response-2",
                output_text="6 times 7 is 42.",
                output=[],
            ),
        ]
    )

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    result = agent.ask("What is 6 times 7?")

    assert result == "6 times 7 is 42."
    assert agent.previous_response_id == "response-2"
    assert len(llm_client.calls) == 2

    second_call = llm_client.calls[1]

    assert second_call["previous_response_id"] == "response-1"
    assert second_call["input_data"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "42",
        }
    ]


def test_agent_handles_multiple_tool_calls() -> None:
    llm_client = FakeLLMClient(
        responses=[
            FakeResponse(
                id="response-1",
                output_text="",
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="calculator",
                        arguments='{"expression": "12 + 8"}',
                        call_id="call-1",
                    )
                ],
            ),
            FakeResponse(
                id="response-2",
                output_text="",
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="calculator",
                        arguments='{"expression": "20 * 5"}',
                        call_id="call-2",
                    )
                ],
            ),
            FakeResponse(
                id="response-3",
                output_text="The result is 100.",
                output=[],
            ),
        ]
    )

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    result = agent.ask("Add 12 and 8, then multiply the result by 5.")

    assert result == "The result is 100."
    assert agent.previous_response_id == "response-3"
    assert len(llm_client.calls) == 3

    assert llm_client.calls[1]["input_data"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "20",
        }
    ]

    assert llm_client.calls[2]["input_data"] == [
        {
            "type": "function_call_output",
            "call_id": "call-2",
            "output": "100",
        }
    ]


def test_agent_raises_error_for_unknown_tool() -> None:
    llm_client = FakeLLMClient(
        responses=[
            FakeResponse(
                id="response-1",
                output_text="",
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="unknown_tool",
                        arguments="{}",
                        call_id="call-1",
                    )
                ],
            )
        ]
    )

    agent = Agent(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
    )

    with pytest.raises(
        ValueError,
        match="Unknown tool: unknown_tool",
    ):
        agent.ask("Use an unknown tool.")


def test_agent_raises_error_when_max_steps_is_reached() -> None:
    responses = [
        FakeResponse(
            id=f"response-{step}",
            output_text="",
            output=[
                FakeFunctionCall(
                    type="function_call",
                    name="calculator",
                    arguments='{"expression": "1 + 1"}',
                    call_id=f"call-{step}",
                )
            ],
        )
        for step in range(1, Agent.MAX_STEPS + 2)
    ]

    llm_client = FakeLLMClient(responses=responses)

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    with pytest.raises(RuntimeError, match="Agent reached the maximum of 5 steps."):
        agent.ask("Keep calculating.")

    assert len(llm_client.calls) == Agent.MAX_STEPS + 1

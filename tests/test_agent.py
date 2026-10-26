import pytest

from dataclasses import dataclass
from typing import Any

from learning_agent.agent import Agent
from learning_agent.tool_registry import ToolRegistry
from learning_agent.tools import CalculatorTool
from learning_agent.memory import SessionMemory
from learning_agent.tools.forget import ForgetTool
from learning_agent.tools.get_memory import GetMemoryTool
from learning_agent.tools.remember import RememberTool


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
        tool_choice: str | dict[str, Any] | None = None,
    ) -> FakeResponse:
        self.calls.append(
            {
                "input_data": input_data,
                "instructions": instructions,
                "tools": tools,
                "previous_response_id": previous_response_id,
                "tool_choice": tool_choice,
            }
        )

        return self.responses.pop(0)


class FailingTool:
    @property
    def name(self) -> str:
        return "failing_tool"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": "A tool that fails during execution.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        raise RuntimeError("Simulated tool failure")


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


def test_agent_can_reset_conversation() -> None:
    llm_client = FakeLLMClient(
        responses=[
            FakeResponse(
                id="response-1",
                output_text="First answer.",
                output=[],
            ),
            FakeResponse(
                id="response-2",
                output_text="Follow-up answer.",
                output=[],
            ),
            FakeResponse(
                id="response-3",
                output_text="Answer in a new conversation.",
                output=[],
            ),
        ]
    )

    agent = Agent(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
    )

    agent.ask("First question?")
    agent.ask("Follow-up question?")
    agent.reset_conversation()
    agent.ask("Question in a new conversation?")

    assert llm_client.calls[0]["previous_response_id"] is None
    assert llm_client.calls[1]["previous_response_id"] == "response-1"
    assert llm_client.calls[2]["previous_response_id"] is None
    assert agent.previous_response_id == "response-3"


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


def test_agent_sends_error_for_unknown_tool() -> None:
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
            ),
            FakeResponse(
                id="response-2",
                output_text="I couldn't use that tool.",
                output=[],
            ),
        ]
    )

    agent = Agent(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
    )

    answer = agent.ask("Use an unknown tool.")

    assert answer == "I couldn't use that tool."
    tool_output = llm_client.calls[1]["input_data"][0]
    assert tool_output["type"] == "function_call_output"
    assert tool_output["call_id"] == "call-1"
    assert "Unknown tool: unknown_tool" in tool_output["output"]


def test_agent_sends_error_for_invalid_tool_arguments() -> None:
    llm_client = FakeLLMClient(
        responses=[
            FakeResponse(
                id="response-1",
                output_text="",
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="calculator",
                        arguments="{invalid json",
                        call_id="call-1",
                    )
                ],
            ),
            FakeResponse(
                id="response-2",
                output_text="I couldn't process the calculator arguments.",
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

    answer = agent.ask("Calculate something.")

    assert answer == "I couldn't process the calculator arguments."
    assert len(llm_client.calls) == 2

    tool_output = llm_client.calls[1]["input_data"][0]
    assert tool_output["type"] == "function_call_output"
    assert tool_output["call_id"] == "call-1"
    assert "not valid JSON" in tool_output["output"]


def test_agent_recovers_from_invalid_tool_arguments() -> None:
    llm_client = FakeLLMClient(
        responses=[
            FakeResponse(
                id="response-1",
                output_text="",
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="calculator",
                        arguments="{invalid json",
                        call_id="call-1",
                    )
                ],
            ),
            FakeResponse(
                id="response-2",
                output_text="The result is 42.",
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

    answer = agent.ask("Calculate something.")

    assert answer == "The result is 42."
    assert len(llm_client.calls) == 2
    assert llm_client.calls[1]["previous_response_id"] == "response-1"

    tool_output = llm_client.calls[1]["input_data"][0]
    assert tool_output["type"] == "function_call_output"
    assert tool_output["call_id"] == "call-1"
    assert "not valid JSON" in tool_output["output"]

    assert agent.previous_response_id == "response-2"


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


def test_agent_includes_memory_tool_instructions() -> None:
    llm_client = FakeLLMClient(
        responses=[
            FakeResponse(
                id="response-1",
                output_text="I can help with that.",
                output=[],
            )
        ]
    )

    agent = Agent(
        llm_client=llm_client,
        tool_registry=ToolRegistry(),
    )

    agent.ask("Hello")

    instructions = llm_client.calls[0]["instructions"]

    assert "remember tool" in instructions
    assert "get_memory tool" in instructions
    assert "forget tool" in instructions
    assert "unless the corresponding tool confirms the operation" in instructions


def test_agent_does_not_require_a_tool_for_general_questions() -> None:
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
    assert len(llm_client.calls) == 1


def test_agent_forces_remember_tool_for_explicit_memory_request() -> None:
    llm_client = FakeLLMClient(
        responses=[
            FakeResponse(
                id="response-1",
                output_text="",
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="remember",
                        arguments='{"key": "language", "value": "Python"}',
                        call_id="call-1",
                    )
                ],
            ),
            FakeResponse(
                id="response-2",
                output_text="I have remembered that you prefer Python.",
                output=[],
            ),
        ]
    )

    memory = SessionMemory()
    tool_registry = ToolRegistry()
    tool_registry.register(RememberTool(memory))

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    result = agent.ask("Remember that I prefer Python.")

    assert result == "I have remembered that you prefer Python."
    assert memory.get("language") == "Python"
    assert llm_client.calls[0]["tool_choice"] == {
        "type": "function",
        "name": "remember",
    }
    assert llm_client.calls[1]["tool_choice"] is None


def test_agent_does_not_force_a_tool_for_general_questions() -> None:
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
    assert llm_client.calls[0]["tool_choice"] is None


def test_agent_forces_forget_tool_for_explicit_memory_request() -> None:
    llm_client = FakeLLMClient(
        responses=[
            FakeResponse(
                id="response-1",
                output_text="",
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="forget",
                        arguments='{"key": "language"}',
                        call_id="call-1",
                    )
                ],
            ),
            FakeResponse(
                id="response-2",
                output_text="I have forgotten your preferred language.",
                output=[],
            ),
        ]
    )

    memory = SessionMemory()
    memory.remember("language", "Python")

    tool_registry = ToolRegistry()
    tool_registry.register(ForgetTool(memory))

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    result = agent.ask("Forget my preferred language.")

    assert result == "I have forgotten your preferred language."
    assert memory.get("language") is None
    assert llm_client.calls[0]["tool_choice"] == {
        "type": "function",
        "name": "forget",
    }


def test_agent_forces_get_memory_tool_for_retrieval_request() -> None:
    llm_client = FakeLLMClient(
        responses=[
            FakeResponse(
                id="response-1",
                output_text="",
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="get_memory",
                        arguments='{"key": "language"}',
                        call_id="call-1",
                    )
                ],
            ),
            FakeResponse(
                id="response-2",
                output_text="You told me that you prefer Python.",
                output=[],
            ),
        ]
    )

    memory = SessionMemory()
    memory.remember("language", "Python")

    tool_registry = ToolRegistry()
    tool_registry.register(GetMemoryTool(memory))

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    result = agent.ask("What did I tell you about my preferred language?")

    assert result == "You told me that you prefer Python."
    assert memory.get("language") == "Python"
    assert llm_client.calls[0]["tool_choice"] == {
        "type": "function",
        "name": "get_memory",
    }


def test_agent_sends_error_when_tool_execution_fails() -> None:
    llm_client = FakeLLMClient(
        responses=[
            FakeResponse(
                id="response-1",
                output_text="",
                output=[
                    FakeFunctionCall(
                        type="function_call",
                        name="failing_tool",
                        arguments="{}",
                        call_id="call-1",
                    )
                ],
            ),
            FakeResponse(
                id="response-2",
                output_text="The tool failed, so I couldn't complete that.",
                output=[],
            ),
        ]
    )

    tool_registry = ToolRegistry()
    tool_registry.register(FailingTool())

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    answer = agent.ask("Use the failing tool.")

    assert answer == "The tool failed, so I couldn't complete that."
    tool_output = llm_client.calls[1]["input_data"][0]
    assert tool_output["type"] == "function_call_output"
    assert tool_output["call_id"] == "call-1"
    assert "Simulated tool failure" in tool_output["output"]

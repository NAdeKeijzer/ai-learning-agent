from typing import Any
from unittest.mock import patch

from learning_agent.llm import LLMClient


def test_create_response_passes_tools_to_openai_api() -> None:
    fake_response = object()

    with patch("learning_agent.llm.OpenAI") as openai_mock:
        client = LLMClient.__new__(LLMClient)
        client.model = "test-model"
        client.client = openai_mock.return_value
        client.client.responses.create.return_value = fake_response

        result = client.create_response(
            input_data="Remember my preferred language is Dutch.",
            instructions="Use the remember tool when asked to store information.",
            tools=[
                {
                    "type": "function",
                    "name": "remember",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            ],
        )

    assert result is fake_response

    call_args: dict[str, Any] = client.client.responses.create.call_args.kwargs

    assert call_args["model"] == "test-model"
    assert call_args["tools"][0]["name"] == "remember"
    assert call_args["input"] == "Remember my preferred language is Dutch."


def test_create_response_passes_tool_choice_to_openai_api() -> None:
    fake_response = object()

    with patch("learning_agent.llm.OpenAI") as openai_mock:
        client = LLMClient.__new__(LLMClient)
        client.model = "test-model"
        client.client = openai_mock.return_value
        client.client.responses.create.return_value = fake_response

        result = client.create_response(
            input_data="Remember my preferred language is Dutch.",
            instructions="Use the remember tool when asked to store information.",
            tools=[],
            tool_choice={
                "type": "function",
                "name": "remember",
            },
        )

    assert result is fake_response

    call_args = client.client.responses.create.call_args.kwargs
    assert call_args["tool_choice"] == {
        "type": "function",
        "name": "remember",
    }

from typing import Any

from learning_agent.main import create_agent
from learning_agent.tools import CalculatorTool, WordCountTool
from learning_agent.tools.forget import ForgetTool
from learning_agent.tools.get_memory import GetMemoryTool
from learning_agent.tools.remember import RememberTool


class FakeLLMClient:
    """A test double that avoids making real OpenAI API calls."""

    def create_response(self, **kwargs: Any) -> Any:
        raise NotImplementedError("This test client should not make API calls.")


def test_create_agent_registers_all_tools() -> None:
    agent = create_agent(llm_client=FakeLLMClient())  # type: ignore[arg-type]

    assert isinstance(agent.tool_registry.get("calculator"), CalculatorTool)
    assert isinstance(agent.tool_registry.get("word_count"), WordCountTool)
    assert isinstance(agent.tool_registry.get("remember"), RememberTool)
    assert isinstance(agent.tool_registry.get("get_memory"), GetMemoryTool)
    assert isinstance(agent.tool_registry.get("forget"), ForgetTool)


def test_memory_tools_share_the_same_session_memory() -> None:
    agent = create_agent(llm_client=FakeLLMClient())  # type: ignore[arg-type]

    remember_tool = agent.tool_registry.get("remember")
    get_memory_tool = agent.tool_registry.get("get_memory")
    forget_tool = agent.tool_registry.get("forget")

    assert remember_tool.memory is get_memory_tool.memory
    assert get_memory_tool.memory is forget_tool.memory

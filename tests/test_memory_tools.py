import pytest

from learning_agent.exceptions.memory import SessionMemoryError
from learning_agent.memory import SessionMemory
from learning_agent.tools.remember import RememberTool
from learning_agent.tools.get_memory import GetMemoryTool
from learning_agent.tools.forget import ForgetTool


def test_remember_tool_stores_value_in_session_memory():
    memory = SessionMemory()
    tool = RememberTool(memory)

    result = tool.execute({"key": "preferred_language", "value": "Dutch"})

    assert result == "Remembered 'preferred_language'."
    assert memory.get("preferred_language") == "Dutch"


def test_get_memory_tool_returns_stored_value():
    memory = SessionMemory()
    memory.remember("preferred_language", "Dutch")
    tool = GetMemoryTool(memory)

    result = tool.execute({"key": "preferred_language"})

    assert result == "Dutch"


def test_remember_tool_raises_error_when_key_already_exists():
    memory = SessionMemory()
    memory.remember("preferred_language", "Dutch")
    tool = RememberTool(memory)

    with pytest.raises(SessionMemoryError):
        tool.execute({"key": "preferred_language", "value": "English"})

    assert memory.get("preferred_language") == "Dutch"


def test_get_memory_tool_returns_message_when_key_does_not_exist():
    memory = SessionMemory()
    tool = GetMemoryTool(memory)

    result = tool.execute({"key": "preferred_language"})

    assert result == "No memory found for key 'preferred_language'."


def test_forget_tool_removes_value_from_session_memory():
    memory = SessionMemory()
    memory.remember("preferred_language", "Dutch")
    tool = ForgetTool(memory)

    result = tool.execute({"key": "preferred_language"})

    assert result == "Forgot 'preferred_language'."
    assert memory.get("preferred_language") is None


def test_forget_tool_raises_error_when_key_does_not_exist():
    memory = SessionMemory()
    tool = ForgetTool(memory)

    with pytest.raises(SessionMemoryError):
        tool.execute({"key": "preferred_language"})

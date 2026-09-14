import pytest

from learning_agent.exceptions.memory import SessionMemoryError
from learning_agent.memory import SessionMemory


def test_remember_stores_new_key_value_pair():
    memory = SessionMemory()

    memory.remember("name", "Nikita")

    assert memory.get("name") == "Nikita"


def test_get_returns_none_for_unknown_key():
    memory = SessionMemory()

    assert memory.get("unknown") is None


def test_items_returns_stored_items():
    memory = SessionMemory()

    memory.remember("name", "Nikita")

    assert memory.items == {"name": "Nikita"}


def test_items_returns_empty_dict_for_empty_memory():
    memory = SessionMemory()

    assert memory.items == {}


def test_remember_raises_error_for_duplicate_key():
    memory = SessionMemory()

    memory.remember("name", "Nikita")

    with pytest.raises(SessionMemoryError):
        memory.remember("name", "John")


def test_forget_removes_existing_key():
    memory = SessionMemory()

    memory.remember("name", "Nikita")
    memory.forget("name")

    assert memory.get("name") is None


def test_forget_raises_error_for_unknown_key():
    memory = SessionMemory()

    with pytest.raises(SessionMemoryError):
        memory.forget("unknown")


def test_modifying_items_does_not_modify_memory():
    memory = SessionMemory()

    memory.remember("name", "Nikita")

    items = memory.items
    items["name"] = "John"

    assert memory.get("name") == "Nikita"

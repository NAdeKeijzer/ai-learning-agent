from learning_agent.exceptions.memory import SessionMemoryError


class SessionMemory:
    def __init__(self) -> None:
        self._items: dict[str, str] = {}

    def remember(self, key: str, value: str) -> None:
        if key in self._items:
            raise SessionMemoryError(f"Memory key already exists: {key}")

        self._items[key] = value

    def get(self, key: str) -> str | None:
        return self._items.get(key)

    def forget(self, key: str) -> None:
        if key not in self._items:
            raise SessionMemoryError(f"Memory key does not exist: {key}")

        del self._items[key]

    @property
    def items(self) -> dict[str, str]:
        return self._items.copy()

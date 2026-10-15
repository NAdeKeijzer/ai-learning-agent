from learning_agent.tool_router import ToolRouter


def test_routes_remember_request() -> None:
    router = ToolRouter()

    assert router.route("Remember that my preferred language is Dutch.") == {
        "type": "function",
        "name": "remember",
    }


def test_routes_memory_retrieval_request() -> None:
    router = ToolRouter()

    assert router.route("What did I tell you about my preferred language?") == {
        "type": "function",
        "name": "get_memory",
    }


def test_routes_forget_request() -> None:
    router = ToolRouter()

    assert router.route("Forget my preferred language.") == {
        "type": "function",
        "name": "forget",
    }


def test_does_not_force_tool_for_general_question() -> None:
    router = ToolRouter()

    assert router.route("What is an AI agent?") is None

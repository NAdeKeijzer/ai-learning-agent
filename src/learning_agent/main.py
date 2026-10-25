import logging

from learning_agent.agent import Agent
from learning_agent.llm import LLMClient
from learning_agent.memory import SessionMemory
from learning_agent.tool_registry import ToolRegistry
from learning_agent.tools import CalculatorTool, WordCountTool
from learning_agent.tools.forget import ForgetTool
from learning_agent.tools.get_memory import GetMemoryTool
from learning_agent.tools.remember import RememberTool


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def create_agent(llm_client: LLMClient | None = None) -> Agent:
    """Create an agent with all available tools."""
    memory = SessionMemory()

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(WordCountTool())
    tool_registry.register(RememberTool(memory))
    tool_registry.register(GetMemoryTool(memory))
    tool_registry.register(ForgetTool(memory))

    return Agent(
        llm_client=llm_client if llm_client is not None else LLMClient(),
        tool_registry=tool_registry,
    )


def main() -> None:
    configure_logging()
    agent = create_agent()

    print("AI learning agent")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        answer = agent.ask(question)
        print(f"\nAI: {answer}\n")


if __name__ == "__main__":
    main()

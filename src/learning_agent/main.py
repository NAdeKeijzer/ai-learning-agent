import logging

from learning_agent.agent import Agent
from learning_agent.llm import LLMClient
from learning_agent.tool_registry import ToolRegistry
from learning_agent.tools import CalculatorTool, WordCountTool


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()

    llm_client = LLMClient()

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(WordCountTool())

    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    print("AI learning agent")
    print("Typ 'exit' om te stoppen.\n")

    while True:
        question = input("Jij: ").strip()

        if question.lower() == "exit":
            print("Tot ziens!")
            break

        if not question:
            continue

        answer = agent.ask(question)

        print(f"\nAI: {answer}\n")


if __name__ == "__main__":
    main()

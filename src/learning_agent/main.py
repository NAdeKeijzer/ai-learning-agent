import logging

from learning_agent.agent import Agent
from learning_agent.llm import LLMClient


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()

    llm_client = LLMClient()
    agent = Agent(llm_client)

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

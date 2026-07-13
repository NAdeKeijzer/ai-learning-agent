import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ["OPENAI_MODEL"]

    client = OpenAI(api_key=api_key)
    previous_response_id: str | None = None

    instructions = (
        "Je bent een behulpzame docent die AI-concepten helder en "
        "beknopt in het Nederlands uitlegt."
    )

    print("AI learning agent")
    print("Typ 'exit' om te stoppen.\n")

    while True:
        user_input = input("Jij: ").strip()

        if user_input.lower() == "exit":
            print("Tot ziens!")
            break

        if not user_input:
            continue

        if previous_response_id is None:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=user_input,
            )
        else:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=user_input,
                previous_response_id=previous_response_id,
            )

        # Deze regels horen binnen while, maar buiten het if/else-blok.
        print(f"\nAI: {response.output_text}\n")
        previous_response_id = response.id


if __name__ == "__main__":
    main()

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    env_file = project_root / ".env"

    loaded = load_dotenv(dotenv_path=env_file)

    print(f".env-pad: {env_file}")
    print(f".env bestaat: {env_file.exists()}")
    print(f".env geladen: {loaded}")

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY ontbreekt. Voeg deze toe aan het .env-bestand."
        )

    if not model:
        raise RuntimeError(
            "OPENAI_MODEL ontbreekt. Voeg deze toe aan het .env-bestand."
        )

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        input="Leg in maximaal drie zinnen uit wat een AI-agent is.",
    )

    print(response.output_text)


if __name__ == "__main__":
    main()
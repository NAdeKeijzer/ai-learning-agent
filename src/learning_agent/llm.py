import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import Response


logger = logging.getLogger("LLMClient")


class LLMClient:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")

        api_key = os.environ["OPENAI_API_KEY"]

        self.model = os.environ["OPENAI_MODEL"]
        self.client = OpenAI(api_key=api_key)

    def create_response(
        self,
        *,
        input_data: str | list[dict[str, str]],
        instructions: str,
        tools: list[dict[str, Any]],
        previous_response_id: str | None = None,
    ) -> Response:
        logger.info(
            "Response aanvragen (model=%s, vervolg=%s, input=%s)",
            self.model,
            previous_response_id is not None,
            type(input_data).__name__,
        )

        if previous_response_id is None:
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                tools=tools,
                input=input_data,
            )
        else:
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                tools=tools,
                input=input_data,
                previous_response_id=previous_response_id,
            )

        logger.info("Response ontvangen")

        return response

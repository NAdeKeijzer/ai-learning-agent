class ToolRouter:
    def route(self, question: str) -> dict[str, str] | None:
        normalized_question = question.lower().strip()

        if normalized_question.startswith("remember "):
            return {
                "type": "function",
                "name": "remember",
            }

        if normalized_question.startswith("forget "):
            return {
                "type": "function",
                "name": "forget",
            }

        if normalized_question.startswith(
            "what did i tell you"
        ) or normalized_question.startswith("what have i told you"):
            return {
                "type": "function",
                "name": "get_memory",
            }

        return None

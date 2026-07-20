import ast
import logging
import operator
from typing import Any


logger = logging.getLogger("Calculator")

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorTool:
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Voert een rekenkundige berekening uit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "De rekenkundige expressie.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        expression = arguments["expression"]

        logger.info("Berekening uitvoeren: %s", expression)

        parsed_expression = ast.parse(expression, mode="eval")
        result = self._evaluate(parsed_expression.body)

        return str(result)

    def _evaluate(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value

        if isinstance(node, ast.BinOp):
            operator_function = _OPERATORS.get(type(node.op))

            if operator_function is None:
                raise ValueError("Niet-ondersteunde rekenkundige operator.")

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            return operator_function(left, right)

        if isinstance(node, ast.UnaryOp):
            operator_function = _OPERATORS.get(type(node.op))

            if operator_function is None:
                raise ValueError("Niet-ondersteunde rekenkundige operator.")

            return operator_function(self._evaluate(node.operand))

        raise ValueError("Ongeldige rekenkundige expressie.")

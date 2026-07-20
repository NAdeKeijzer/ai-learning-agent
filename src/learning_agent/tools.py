import logging


logger = logging.getLogger("Calculator")


def calculate(a: float, b: float, operation: str) -> float:
    logger.info(
        "Berekening uitvoeren: a=%s, b=%s, operation=%s",
        a,
        b,
        operation,
    )

    match operation:
        case "add":
            result = a + b
        case "subtract":
            result = a - b
        case "multiply":
            result = a * b
        case "divide":
            if b == 0:
                raise ValueError("Delen door nul is niet toegestaan.")
            result = a / b
        case _:
            raise ValueError(f"Onbekende bewerking: {operation}")

    logger.info("Berekening afgerond: resultaat=%s", result)

    return result

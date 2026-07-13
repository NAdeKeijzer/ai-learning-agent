def calculate(a: float, b: float, operation: str) -> float:
    match operation:
        case "add":
            return a + b
        case "subtract":
            return a - b
        case "multiply":
            return a * b
        case "divide":
            if b == 0:
                raise ValueError("Delen door nul is niet toegestaan.")
            return a / b
        case _:
            raise ValueError(f"Onbekende bewerking: {operation}")

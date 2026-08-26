import pytest

from learning_agent.tools import CalculatorTool


@pytest.fixture
def calculator() -> CalculatorTool:
    return CalculatorTool()


def test_addition(calculator: CalculatorTool) -> None:
    result = calculator.execute({"expression": "12 + 8"})

    assert result == "20"


def test_subtraction(calculator: CalculatorTool) -> None:
    result = calculator.execute({"expression": "12 - 8"})

    assert result == "4"


def test_multiplication(calculator: CalculatorTool) -> None:
    result = calculator.execute({"expression": "6 * 7"})

    assert result == "42"


def test_division(calculator: CalculatorTool) -> None:
    result = calculator.execute({"expression": "12 / 3"})

    assert result == "4.0"


def test_floor_division(calculator: CalculatorTool) -> None:
    result = calculator.execute({"expression": "10 // 3"})

    assert result == "3"


def test_modulo(calculator: CalculatorTool) -> None:
    result = calculator.execute({"expression": "10 % 3"})

    assert result == "1"


def test_power(calculator: CalculatorTool) -> None:
    result = calculator.execute({"expression": "2 ** 3"})

    assert result == "8"


def test_negative_number(calculator: CalculatorTool) -> None:
    result = calculator.execute({"expression": "-5 + 2"})

    assert result == "-3"


def test_positive_unary_operator(calculator: CalculatorTool) -> None:
    result = calculator.execute({"expression": "+5 + 2"})

    assert result == "7"


def test_parentheses(calculator: CalculatorTool) -> None:
    result = calculator.execute({"expression": "(12 + 8) * 5"})

    assert result == "100"


def test_division_by_zero_raises_error(calculator: CalculatorTool) -> None:
    with pytest.raises(ZeroDivisionError):
        calculator.execute({"expression": "10 / 0"})


def test_function_call_is_rejected(calculator: CalculatorTool) -> None:
    with pytest.raises(
        ValueError,
        match="Invalid arithmetic expression",
    ):
        calculator.execute({"expression": "print('hello')"})


def test_variable_is_rejected(calculator: CalculatorTool) -> None:
    with pytest.raises(
        ValueError,
        match="Invalid arithmetic expression",
    ):
        calculator.execute({"expression": "some_variable + 1"})

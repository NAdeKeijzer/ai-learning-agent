import pytest

from learning_agent.tools import WordCountTool


@pytest.fixture
def word_count_tool() -> WordCountTool:
    return WordCountTool()


def test_counts_words(word_count_tool: WordCountTool) -> None:
    result = word_count_tool.execute({"text": "AI agents can use multiple tools"})

    assert result == "6"


def test_empty_text_contains_zero_words(
    word_count_tool: WordCountTool,
) -> None:
    result = word_count_tool.execute({"text": ""})

    assert result == "0"


def test_single_word(word_count_tool: WordCountTool) -> None:
    result = word_count_tool.execute({"text": "Kafka"})

    assert result == "1"


def test_ignores_multiple_spaces(
    word_count_tool: WordCountTool,
) -> None:
    result = word_count_tool.execute({"text": "AI   agents   use   tools"})

    assert result == "4"


def test_handles_newlines_and_tabs(
    word_count_tool: WordCountTool,
) -> None:
    result = word_count_tool.execute({"text": "AI\nagents\tuse tools"})

    assert result == "4"


def test_leading_and_trailing_whitespace(
    word_count_tool: WordCountTool,
) -> None:
    result = word_count_tool.execute({"text": "   AI agents use tools   "})

    assert result == "4"

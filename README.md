# AI Learning Agent

A small Python project for learning how AI agents work by building one step by step.

The project uses the OpenAI Responses API and focuses on understanding the underlying concepts behind agentic applications, such as tool calling, conversation state, tool registration, testing, and continuous integration.

## Features

The agent currently supports:

- Interactive command-line conversations
- Conversation state between questions
- Tool calling through the OpenAI Responses API
- Multiple tool calls across agent steps
- A configurable tool registry
- Calculator tool for arithmetic expressions
- Word count tool
- Structured application logging
- Automated unit tests with pytest
- Code formatting and linting with Ruff
- Continuous integration with GitHub Actions

## Project Structure

```text
ai-learning-agent/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── learning_agent/
│       ├── __init__.py
│       ├── agent.py
│       ├── llm.py
│       ├── main.py
│       ├── tool_registry.py
│       └── tools/
│           ├── __init__.py
│           ├── calculator.py
│           └── word_count.py
├── tests/
│   ├── test_agent.py
│   ├── test_calculator.py
│   ├── test_tool_registry.py
│   └── test_word_count.py
├── .env.example
├── .gitignore
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key

## Installation

Clone the repository:

```bash
git clone https://github.com/NAdeKeijzer/ai-learning-agent.git
cd ai-learning-agent
```

Install the project and its dependencies:

```bash
uv sync
```

## Configuration

Create a `.env` file in the project root based on `.env.example`.

```text
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=your-model
```

The `.env` file is ignored by Git and should never be committed.

## Running the Agent

Start the application with:

```bash
uv run ai-learning-agent
```

You can then interact with the agent from the command line:

```text
AI learning agent
Type 'exit' to quit.

You: What is Kafka?

AI: ...

You: What is 6 times 7?

AI: 6 times 7 is 42.
```

The agent decides whether it can answer a question directly or whether one of its registered tools should be used.

## Tools

Tools are registered through `ToolRegistry`.

This keeps tool selection and execution separate from the main agent logic and makes it easier to add additional tools later.

### Calculator

The calculator evaluates arithmetic expressions such as:

```text
12 + 8
6 * 7
(12 + 8) * 5
2 ** 3
```

The calculator uses Python's abstract syntax tree rather than evaluating arbitrary Python code.

### Word Count

The word count tool counts the number of words in a supplied text.

## Testing

The project uses pytest for automated testing.

Run all tests with:

```bash
uv run pytest -v
```

The current test suite covers:

- Agent responses without tool calls
- Single and sequential tool calls
- Agent error handling and maximum step limits
- Calculator operations and invalid expressions
- Word counting
- Tool registration and lookup
- Duplicate and unknown tool handling
- Tool definitions

## Code Quality

Check formatting:

```bash
uv run ruff format --check .
```

Format the code:

```bash
uv run ruff format .
```

Run linting:

```bash
uv run ruff check .
```

## Continuous Integration

GitHub Actions automatically runs quality checks for pushes and pull requests targeting `main`.

The CI pipeline:

1. Checks out the repository
2. Installs uv and Python
3. Installs the locked dependencies
4. Checks formatting with Ruff
5. Runs Ruff linting
6. Runs the pytest test suite

This ensures that changes are automatically verified on a clean Linux environment before they are merged.

## Development Workflow

Development is done through short-lived branches rather than directly on `main`.

A typical change follows this workflow:

```text
main
  ↓
feature/chore branch
  ↓
development
  ↓
local Ruff + pytest checks
  ↓
commit
  ↓
push
  ↓
pull request
  ↓
GitHub Actions
  ↓
merge to main
```

## Learning Goals

This repository is primarily a learning project. The goal is to gradually explore concepts involved in building AI agents while maintaining good software engineering practices.

Topics include:

- OpenAI Responses API
- Tool calling
- Agent loops
- Conversation state
- Dependency injection
- Extensible tool architecture
- Automated testing
- CI/CD
- Git and pull request workflows

Future iterations may explore more advanced agent capabilities and additional tools.
# CONSTRIBUTING

This file provides guidance for working with code in this repository.

## Build/Test Commands
- Install package: `pip install -e ".[dev]"`
- Run all tests: `pytest`
- Run single test: `pytest tests/test_claude_tools.py::TestCase::test_tools`
- Type checking: `mypy claude_llm_tools`
- Lint check: `ruff check claude_llm_tools tests`
- Format code: `ruff format claude_llm_tools tests`

## Style Guidelines
- Python 3.12+ with type hints
- Line length: 79 characters max (PEP-8)
- Quote style: Single quotes for strings
- Use docstrings for all functions/classes
- Import order: std lib, third-party, local modules
- Avoid bare except clauses; catch specific exceptions
- Class naming: CapWords (PascalCase)
- Function/variable naming: snake_case
- Constants: UPPERCASE
- Use Pydantic for data models
- Type hints: Use Python 3.10+ union operator (`|` not `Union`)
- Error handling: Create appropriate error results using `error_result`

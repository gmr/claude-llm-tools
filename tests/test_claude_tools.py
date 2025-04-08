import asyncio
import unittest
from unittest import mock

import jsonschema_models
from anthropic import types

import claude_llm_tools
from claude_llm_tools import models, state


@claude_llm_tools.tool
def add_numbers(_req: claude_llm_tools.Request, a: int, b: int) -> int:
    """
    Add two numbers together and return the result.

    Args:
        a: The first number
        b: The second number

    Returns:
        The sum of a and b
    """
    return a + b


@claude_llm_tools.tool
def multiply_numbers(_req: claude_llm_tools.Request, a: int, b: int) -> int:
    """
    Multiply two numbers together and return the result.

    Args:
        a: The first number
        b: The second number

    Returns:
        The product of a and b
    """
    return a * b


class TestCase(unittest.TestCase):
    maxDiff = 10000000000

    def test_tools_registered(self):
        self.assertEqual(state.get_tool('add_numbers').callable, add_numbers)
        self.assertEqual(
            state.get_tool('multiply_numbers').callable, multiply_numbers
        )

    def test_unknown_tool_raises(self):
        with self.assertRaises(ValueError):
            state.get_tool('unknown-tool')

    def test_tools(self):
        expectation = [
            {
                'description': 'Add two numbers together and return the result'
                '.\n\nArgs:\n    a: The first number\n    b: Th'
                'e second number\n\nReturns:\n    The sum of a '
                'and b',
                'input_schema': {
                    'properties': {
                        'a': {'title': 'A', 'type': 'integer'},
                        'b': {'title': 'B', 'type': 'integer'},
                    },
                    'required': ['a', 'b'],
                    'type': 'object',
                },
                'name': 'add_numbers',
            },
            {
                'description': 'Multiply two numbers together and return the '
                'result.\n\nArgs:\n    a: The first number\n  '
                '  b: The second number\n\nReturns:\n    The '
                'product of a and b',
                'input_schema': {
                    'properties': {
                        'a': {'title': 'A', 'type': 'integer'},
                        'b': {'title': 'B', 'type': 'integer'},
                    },
                    'required': ['a', 'b'],
                    'type': 'object',
                },
                'name': 'multiply_numbers',
            },
        ]
        self.assertEqual(claude_llm_tools.tools(), expectation)


class TestAdditionalCoverage(unittest.TestCase):
    def setUp(self):
        # Save the original tools registry
        self._original_tools = state.State.get_instance().tools.copy()
        # Clear the tools registry before each test
        state.State.get_instance().tools = []

    def tearDown(self):
        # Restore the original tools registry
        state.State.get_instance().tools = self._original_tools

    def test_add_tool(self):
        """Test manually adding a tool."""

        def sample_function(
            request: claude_llm_tools.Request, param1: str
        ) -> str:
            """Sample function docstring."""
            return f'Processed: {param1}'

        # Add the tool manually
        claude_llm_tools.add_tool(sample_function)

        # Verify the tool was added
        tool = state.get_tool('sample_function')
        self.assertEqual(tool.name, 'sample_function')
        self.assertEqual(tool.callable, sample_function)
        self.assertEqual(tool.description, 'Sample function docstring.')
        self.assertIsNotNone(tool.input_schema)

    def test_add_tool_with_custom_name(self):
        """Test manually adding a tool with a custom name."""

        def sample_function(
            request: claude_llm_tools.Request, param1: str
        ) -> str:
            return f'Processed: {param1}'

        # Add the tool with a custom name
        claude_llm_tools.add_tool(sample_function, name='custom_name')

        # Verify the tool was added with the custom name
        tool = state.get_tool('custom_name')
        self.assertEqual(tool.name, 'custom_name')
        self.assertEqual(tool.callable, sample_function)

    def test_add_tool_with_custom_description(self):
        """Test manually adding a tool with a custom description."""

        def sample_function(
            request: claude_llm_tools.Request, param1: str
        ) -> str:
            return f'Processed: {param1}'

        # Add the tool with a custom description
        custom_description = 'This is a custom description'
        claude_llm_tools.add_tool(
            sample_function, description=custom_description
        )

        # Verify the tool was added with the custom description
        tool = state.get_tool('sample_function')
        self.assertEqual(tool.description, custom_description)

    def test_add_tool_with_custom_input_schema(self):
        """Test manually adding a tool with a custom input schema."""

        def sample_function(
            request: claude_llm_tools.Request, param1: str
        ) -> str:
            return f'Processed: {param1}'

        # Create a custom input schema
        custom_schema = jsonschema_models.Schema(
            type='object',
            properties={
                'param1': {'type': 'string', 'description': 'Custom parameter'}
            },
            required=['param1'],
        )

        # Add the tool with the custom input schema
        claude_llm_tools.add_tool(sample_function, input_schema=custom_schema)

        # Verify the tool was added with the custom input schema
        tool = state.get_tool('sample_function')
        self.assertEqual(tool.input_schema, custom_schema)

    def test_add_tool_with_tool_type(self):
        """Test manually adding a tool with a custom tool type."""

        def sample_function(
            request: claude_llm_tools.Request, param1: str
        ) -> str:
            return f'Processed: {param1}'

        # Add the tool with a custom tool type
        custom_tool_type = 'custom_type'
        claude_llm_tools.add_tool(sample_function, tool_type=custom_tool_type)

        # Verify the tool was added with the custom tool type
        tool = state.get_tool('sample_function')
        self.assertEqual(tool.type, custom_tool_type)
        self.assertIsNone(tool.input_schema)

    def test_error_result(self):
        """Test creating an error result."""
        result = claude_llm_tools.error_result(
            'tool123', 'Something went wrong'
        )

        self.assertEqual(result.tool_use_id, 'tool123')
        self.assertEqual(result.content, 'Error: Something went wrong')
        self.assertTrue(result.is_error)
        self.assertEqual(result.type, 'tool_result')

    def test_success_result(self):
        """Test creating a success result."""
        result = claude_llm_tools.success_result(
            'tool123', 'Operation successful'
        )

        self.assertEqual(result.tool_use_id, 'tool123')
        self.assertEqual(result.content, 'Operation successful')
        self.assertFalse(result.is_error)
        self.assertEqual(result.type, 'tool_result')

    async def async_test_dispatch(self):
        """Test dispatching a tool call."""

        # Define a test tool
        @claude_llm_tools.tool
        async def test_tool(
            request: claude_llm_tools.Request, param: str
        ) -> models.Result:
            return claude_llm_tools.success_result(
                request.tool_use.id,
                f'Tool {request.tool_use.name} called with '
                f'input: {request.tool_use.input}',
            )

        # Create a mock ToolUseBlock
        tool_use = mock.Mock(spec=types.ToolUseBlock)
        tool_use.id = 'tool123'
        tool_use.name = 'test_tool'
        tool_use.input = {'param': 'value'}

        # Dispatch the tool call
        result = await claude_llm_tools.dispatch(tool_use)

        # Verify the result
        self.assertEqual(result['tool_use_id'], 'tool123')
        self.assertEqual(
            result['content'],
            "Tool test_tool called with input: {'param': 'value'}",
        )
        self.assertFalse(result['is_error'])

    async def async_test_dispatch_unknown_tool(self):
        """Test dispatching a call to an unknown tool."""
        # Create a mock ToolUseBlock for an unknown tool
        tool_use = mock.Mock(spec=types.ToolUseBlock)
        tool_use.id = 'tool123'
        tool_use.name = 'unknown_tool'

        # Patch state.get_tool to return None instead of raising ValueError
        with mock.patch('claude_llm_tools.state.get_tool', return_value=None):
            # Dispatch the tool call
            result = await claude_llm_tools.dispatch(tool_use)

            # Verify the result is an error
            self.assertEqual(result['tool_use_id'], 'tool123')
            self.assertTrue(result['is_error'])
            self.assertEqual(
                result['content'], 'Error: Tool unknown_tool not found'
            )

    def test_dispatch(self):
        """Run the async dispatch test."""
        asyncio.run(self.async_test_dispatch())

    def test_dispatch_unknown_tool(self):
        """Run the async dispatch unknown tool test."""
        asyncio.run(self.async_test_dispatch_unknown_tool())

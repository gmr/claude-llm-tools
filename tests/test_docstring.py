import inspect
import unittest

from claude_llm_tools import docstring


def _parse(function) -> docstring.Docstring:
    return docstring.parse(inspect.getdoc(function))


class ParseTestCase(unittest.TestCase):
    def test_empty_docstring(self):
        self.assertEqual(
            docstring.parse(None),
            docstring.Docstring(description=None, arguments={}),
        )
        self.assertEqual(
            docstring.parse(''),
            docstring.Docstring(description=None, arguments={}),
        )

    def test_summary_only(self):
        def function():
            """Do a thing."""

        self.assertEqual(
            _parse(function),
            docstring.Docstring(description='Do a thing.', arguments={}),
        )

    def test_removes_sections_and_parses_arguments(self):
        def function(request, channel, name):
            """Add an emoji reaction to a Slack message.

            Use the ``ts`` value from the message you want to react to.

            Args:
                request: The tool request context.
                channel: The Slack channel ID.
                name: The emoji name without colons.

            Returns:
                True if the reaction was added.

            Raises:
                ValueError: When the channel is invalid.

            """

        self.assertEqual(
            _parse(function),
            docstring.Docstring(
                description=(
                    'Add an emoji reaction to a Slack message.\n\n'
                    'Use the ``ts`` value from the message you want to '
                    'react to.'
                ),
                arguments={
                    'request': 'The tool request context.',
                    'channel': 'The Slack channel ID.',
                    'name': 'The emoji name without colons.',
                },
            ),
        )

    def test_keeps_text_after_removed_sections(self):
        def function(value):
            """Summary.

            Args:
                value: The value.

            Example:
                function(1)

            """

        result = _parse(function)
        self.assertEqual(
            result.description, 'Summary.\n\nExample:\n    function(1)'
        )
        self.assertEqual(result.arguments, {'value': 'The value.'})

    def test_joins_continuation_lines(self):
        def function(query):
            """Search.

            Args:
                query: The search query,
                    which can span
                    several lines.

            """

        self.assertEqual(
            _parse(function).arguments,
            {'query': 'The search query, which can span several lines.'},
        )

    def test_description_on_following_line(self):
        def function(query):
            """Search.

            Args:
                query:
                    The search query.

            """

        self.assertEqual(
            _parse(function).arguments, {'query': 'The search query.'}
        )

    def test_typed_and_starred_arguments(self):
        def function(value, *args, **kwargs):
            """Summary.

            Arguments:
                value (int): The value.
                *args: Extra positional values.
                **kwargs: Extra keyword values.

            """

        self.assertEqual(
            _parse(function).arguments,
            {
                'value': 'The value.',
                'args': 'Extra positional values.',
                'kwargs': 'Extra keyword values.',
            },
        )

    def test_continuation_containing_colon(self):
        def function(timestamp):
            """Summary.

            Args:
                timestamp: The message timestamp,
                    e.g. ts: 1234567890.123456.

            """

        self.assertEqual(
            _parse(function).arguments,
            {
                'timestamp': 'The message timestamp, e.g. ts: '
                '1234567890.123456.'
            },
        )

    def test_only_sections(self):
        def function(value):
            """
            Args:
                value: The value.
            """

        self.assertEqual(
            _parse(function),
            docstring.Docstring(
                description=None, arguments={'value': 'The value.'}
            ),
        )

    def test_joins_wrapped_lines_within_paragraphs(self):
        def function():
            """Add an emoji reaction
            to a Slack message.

            Use this tool to acknowledge a message
            without posting a reply.

            """

        self.assertEqual(
            _parse(function).description,
            'Add an emoji reaction to a Slack message.\n\n'
            'Use this tool to acknowledge a message without posting a reply.',
        )

    def test_keeps_list_items_on_their_own_lines(self):
        def function():
            """Summary.

            Modes:
            - fast, which skips
            validation
            - slow
            1. first
            2) second

            """

        self.assertEqual(
            _parse(function).description,
            'Summary.\n\nModes:\n- fast, which skips validation\n- slow\n'
            '1. first\n2) second',
        )

    def test_keeps_indented_lines_and_section_headers(self):
        def function():
            """Summary.
            Example:
                function(1)
                function(2)
            Done.

            """

        self.assertEqual(
            _parse(function).description,
            'Summary.\nExample:\n    function(1)\n    function(2)\nDone.',
        )

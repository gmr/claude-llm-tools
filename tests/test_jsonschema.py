import json
import typing
import unittest

from claude_llm_tools import jsonschema, models


class TestTypeToJsonSchema(unittest.TestCase):
    """Tests for the _type_to_schema function."""

    def test_none_type(self):
        """Test converting None type to JSON schema."""
        schema = jsonschema._type_to_schema(type(None))
        self.assertEqual(schema.model_dump(), {'type': 'null'})

    def test_basic_types(self):
        """Test converting basic Python types to JSON schema."""
        self.assertEqual(
            jsonschema._type_to_schema(str).model_dump(), {'type': 'string'}
        )
        self.assertEqual(
            jsonschema._type_to_schema(int).model_dump(), {'type': 'integer'}
        )
        self.assertEqual(
            jsonschema._type_to_schema(float).model_dump(), {'type': 'number'}
        )
        self.assertEqual(
            jsonschema._type_to_schema(bool).model_dump(), {'type': 'boolean'}
        )
        self.assertEqual(
            jsonschema._type_to_schema(list).model_dump(), {'type': 'array'}
        )
        self.assertEqual(
            jsonschema._type_to_schema(dict).model_dump(), {'type': 'object'}
        )

    def test_list_type(self):
        """Test converting List types to JSON schema."""
        schema = jsonschema._type_to_schema(list[str])
        self.assertEqual(
            schema.model_dump(), {'type': 'array', 'items': {'type': 'string'}}
        )

        schema = jsonschema._type_to_schema(list[int])
        self.assertEqual(
            schema.model_dump(),
            {'type': 'array', 'items': {'type': 'integer'}},
        )

    def test_dict_type(self):
        """Test converting Dict types to JSON schema."""
        schema = json.loads(
            jsonschema._type_to_schema(dict[str, int]).model_dump_json()
        )
        self.assertEqual(
            schema,
            {'type': 'object', 'additionalProperties': {'type': 'integer'}},
        )

        schema = json.loads(
            jsonschema._type_to_schema(dict[str, str]).model_dump_json()
        )
        self.assertEqual(
            schema,
            {'type': 'object', 'additionalProperties': {'type': 'string'}},
        )

    def test_literal_type(self):
        """Test converting Literal types to JSON schema."""
        schema = jsonschema._type_to_schema(
            typing.Literal['red', 'green', 'blue']
        )
        self.assertEqual(
            schema.model_dump(),
            {'type': 'string', 'enum': ['red', 'green', 'blue']},
        )

        schema = jsonschema._type_to_schema(typing.Literal[1, 2, 3])
        self.assertEqual(
            schema.model_dump(), {'type': 'integer', 'enum': [1, 2, 3]}
        )

    def test_union_type(self):
        """Test converting Union types to JSON schema."""
        schema = jsonschema._type_to_schema(str | int)
        self.assertEqual(schema.model_dump(), {'type': ['string', 'integer']})

    def test_complex_union_type(self):
        """Test converting Union types to JSON schema."""
        schema = jsonschema._type_to_schema(str | int | dict[str, int] | bool)
        self.assertEqual(
            schema.model_dump(),
            {
                'anyOf': [
                    {'type': 'string'},
                    {'type': 'integer'},
                    {
                        'type': 'object',
                        'additionalProperties': {'type': 'integer'},
                    },
                    {'type': 'boolean'},
                ]
            },
        )

    def test_optional_type(self):
        """Test converting Optional types to JSON schema."""
        schema = jsonschema._type_to_schema(str | None)
        self.assertEqual(schema.model_dump(), {'type': ['string', 'null']})

        schema = jsonschema._type_to_schema(int | None)
        self.assertEqual(schema.model_dump(), {'type': ['integer', 'null']})


class TestHandleUnionType(unittest.TestCase):
    """Tests for the _handle_union_type function."""

    def test_optional_type(self):
        """Test handling Optional types."""
        schema = jsonschema._handle_union_type(str | None)
        self.assertEqual(schema.model_dump(), {'type': ['string', 'null']})

        schema = jsonschema._handle_union_type(int | None)
        self.assertEqual(schema.model_dump(), {'type': ['integer', 'null']})

    def test_union_type(self):
        """Test handling Union types."""
        schema = jsonschema._handle_union_type(str | int)
        self.assertEqual(schema.model_dump(), {'type': ['string', 'integer']})

        schema = jsonschema._handle_union_type(str | int | bool)
        self.assertEqual(
            schema.model_dump(), {'type': ['string', 'integer', 'boolean']}
        )


class TestToJsonSchema(unittest.TestCase):
    """Tests for the to_json_schema function."""

    maxDiff = 32768

    def test_simple_function(self):
        """Test converting a simple function to JSON schema."""

        def test_func(a: int, b: str) -> str:
            return f'{a} {b}'

        schema = jsonschema.to_schema(test_func)
        expected = {
            'type': 'object',
            'properties': {
                'a': {'type': 'integer', 'title': 'A'},
                'b': {'type': 'string', 'title': 'B'},
            },
            'required': ['a', 'b'],
        }
        self.assertEqual(schema.model_dump(), expected)

    def test_function_with_optional_params(self):
        """Test converting a function w/ optional parameters to JSON schema."""

        def test_func(a: int, b: str = 'default') -> str:
            return f'{a} {b}'

        schema = jsonschema.to_schema(test_func)
        expected = {
            'type': 'object',
            'properties': {
                'a': {'type': 'integer', 'title': 'A'},
                'b': {'type': 'string', 'title': 'B'},
            },
            'required': ['a'],
        }
        self.assertEqual(schema.model_dump(), expected)

    def test_function_with_request_param(self):
        """Test converting a func with a Request parameter to JSON schema."""

        def test_func(_request: models.Request, a: int, b: str) -> str:
            return f'{a} {b}'

        schema = jsonschema.to_schema(test_func)
        expected = {
            'type': 'object',
            'properties': {
                'a': {'type': 'integer', 'title': 'A'},
                'b': {'type': 'string', 'title': 'B'},
            },
            'required': ['a', 'b'],
        }
        self.assertEqual(schema.model_dump(), expected)

    def test_function_with_complex_types(self):
        """Test converting a function with complex types to JSON schema."""

        def test_func(
            a: list[int],
            b: dict[str, str],
            c: str | None,
            d: int | str,
            e: typing.Literal['red', 'green', 'blue'],
        ) -> dict:
            return {}

        schema = jsonschema.to_schema(test_func)
        expected = {
            'type': 'object',
            'properties': {
                'a': {
                    'type': 'array',
                    'items': {'type': 'integer'},
                    'title': 'A',
                },
                'b': {
                    'type': 'object',
                    'additionalProperties': {'type': 'string'},
                    'title': 'B',
                },
                'c': {'type': ['string', 'null'], 'title': 'C'},
                'd': {'type': ['integer', 'string'], 'title': 'D'},
                'e': {
                    'type': 'string',
                    'enum': ['red', 'green', 'blue'],
                    'title': 'E',
                },
            },
            'required': ['a', 'b', 'c', 'd', 'e'],
        }
        self.assertEqual(schema.model_dump(), expected)

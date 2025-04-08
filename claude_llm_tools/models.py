import typing

import jsonschema_models as jsm
import pydantic
from anthropic import types
from anthropic.types import beta


class Request(pydantic.BaseModel):
    """Model representing a tool call request."""

    context: typing.Any | None = None
    tool_use: types.ToolUseBlock | beta.BetaToolUseBlock


class Result(pydantic.BaseModel):
    """Model representing a tool call result."""

    type: str = 'tool_result'
    tool_use_id: str
    content: str = ''
    is_error: bool = False


class InputSchema(pydantic.BaseModel):
    """Represents the input schema for a tool"""


class Tool(pydantic.BaseModel):
    """Represents a tool definition"""

    name: str
    callable: typing.Callable
    description: str | None = None
    input_schema: jsm.Schema | None = None
    type: str | None = None

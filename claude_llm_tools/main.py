import inspect
import json
import logging
import typing

import jsonschema_models as jsm
from anthropic import types
from anthropic.types import beta as beta_types

from claude_llm_tools import docstring, errors, jsonschema, models, state

LOGGER = logging.getLogger(__name__)


def add_tool(
    function: typing.Callable,
    name: str | None = None,
    description: str | None = None,
    input_schema: jsm.Schema | None = None,
    tool_type: str | None = None,
    response_validator: typing.Callable | None = None,
) -> None:
    """Manually add a tool to use instead of registering it with the decorator

    :param function: The function to call when the tool is invoked
    :param name: The optional tool name
    :param description: The optional tool description, defaulting to the
        docstring without its Args, Returns, Yields, and Raises sections
    :param input_schema: The optional input schema for the tool, defaulting
        to one built from the signature and the docstring's Args section
    :param tool_type: The optional tool type
    :param response_validator: The optional response validator

    """
    name = name or function.__name__
    if not tool_type:
        parsed = docstring.parse(inspect.getdoc(function))
        description = description or parsed.description
        input_schema = input_schema or jsonschema.to_schema(function)
    state.add_tool(
        models.Tool(
            name=name,
            callable=function,
            description=description,
            input_schema=input_schema,
            type=tool_type,
            response_validator=response_validator,
        )
    )


def _result_to_str(value: typing.Any) -> str:
    if hasattr(value, 'model_dump_json'):
        return value.model_dump_json()
    elif isinstance(value, dict):
        return json.dumps(value)
    elif isinstance(value, list):
        return '[{}]'.format(','.join([_result_to_str(v) for v in value]))
    return str(value)


async def dispatch(
    tool_use: types.ToolUseBlock | beta_types.BetaToolUseBlock,
    context: typing.Any | None = None,
) -> types.ToolResultBlockParam:
    """Invoke this with the ToolUseBlock from the LLM to call the tool."""
    LOGGER.debug('Tool Use: %r', tool_use)
    request = models.Request(tool_use=tool_use, context=context)
    obj = state.get_tool(request.tool_use.name)
    if not obj:
        return types.ToolResultBlockParam(
            type='tool_result',
            tool_use_id=tool_use.id,
            content=f'Error: Tool {tool_use.name} not found',
            is_error=True,
        )
    kwargs = tool_use.input if tool_use.input else {}
    try:
        result = await obj.callable(request, **kwargs)  # type: ignore
    except (errors.ToolError, TypeError) as err:
        return types.ToolResultBlockParam(
            type='tool_result',
            tool_use_id=tool_use.id,
            content=f'Exception raised: {str(err)}',
            is_error=True,
        )
    if obj.response_validator:
        try:
            result = obj.response_validator(request, result)
        except (errors.ValidationError, ValueError, TypeError) as err:
            return types.ToolResultBlockParam(
                type='tool_result',
                tool_use_id=tool_use.id,
                content=f'Exception raised: {str(err)}',
                is_error=True,
            )
    return types.ToolResultBlockParam(
        type='tool_result',
        tool_use_id=tool_use.id,
        content=_result_to_str(result),
        is_error=False,
    )


def tool[Function: typing.Callable[..., typing.Any]](
    function: Function | None = None,
    name: str | None = None,
    description: str | None = None,
    input_schema: jsm.Schema | None = None,
    tool_type: str | None = None,
    response_validator: typing.Callable | None = None,
) -> typing.Callable[[Function], Function] | Function:
    """Decorator that registers a function as a tool.

    Can be used as a simple decorator or with arguments:

    @tool
    def my_function(): ...

    @tool(name='custom_name', description='Custom description')
    def my_function(): ...
    """

    def decorator(func: Function) -> Function:
        add_tool(
            func,
            name=name,
            description=description,
            input_schema=input_schema,
            tool_type=tool_type,
            response_validator=response_validator,
        )
        return typing.cast(Function, func)

    if function is not None:
        return decorator(function)

    return decorator


def tools() -> list[dict]:
    """Return a list of the installed tools for use when invoking API calls
    to the LLM.

    """
    return [entry.model_dump() for entry in state.get_tools()]

import inspect
import json
import logging
import typing

import jsonschema_models as jsm
from anthropic import types

from claude_llm_tools import jsonschema, models, state

LOGGER = logging.getLogger(__name__)


def add_tool(
    function: typing.Callable,
    name: str | None = None,
    description: str | None = None,
    input_schema: jsm.Schema | None = None,
    tool_type: str | None = None,
) -> None:
    """Manually add a tool to use instead of registering it with the decorator

    :param function: The function to call when the tool is invoked
    :param name: The optional tool name
    :param description: The optional tool description
    :param input_schema: The optional input schema for the tool
    :param tool_type: The optional tool type

    """
    name = name or function.__name__
    if not tool_type:
        description = description or inspect.getdoc(function)
        schema = jsonschema.to_schema(function)
        input_schema = input_schema or jsm.Schema.model_validate(schema)
    state.add_tool(
        models.Tool(
            name=name,
            callable=function,
            description=description,
            input_schema=input_schema,
            type=tool_type,
        )
    )


Function = typing.TypeVar('Function', bound=typing.Callable[..., typing.Any])


def tool(function: Function) -> Function:
    """Decorator that registers a function as a tool."""
    add_tool(function)
    return typing.cast(Function, function)


async def dispatch(
    tool_use: types.ToolUseBlock, context: typing.Any | None = None
) -> dict:
    """Invoke this with the ToolUseBlock from the LLM to call the tool."""
    LOGGER.debug('Tool Use: %r', tool_use)
    request = models.Request(tool_use=tool_use, context=context)
    obj = state.get_tool(request.tool_use.name)
    if not obj:
        return error_result(
            tool_use.id, f'Tool {tool_use.name} not found'
        ).model_dump()
    kwargs = tool_use.input if tool_use.input else {}
    try:
        result = await obj.callable(request, **kwargs)  # type: ignore
    except TypeError as err:
        return error_result(
            tool_use.id, f'Exception raised: {str(err)}'
        ).model_dump()
    return result.model_dump()


def tools() -> list[dict]:
    """Return a list of the installed tools for use when invoking API calls
    to the LLM.

    """
    result = []
    for entry in state.get_tools():
        tool_dict = json.loads(
            entry.model_dump_json(
                exclude={'callable'}, exclude_none=True, by_alias=True
            )
        )
        result.append(tool_dict)
    return result


def error_result(tool_use_id: str, message: str) -> models.Result:
    """Set the error message and return the result as a dict."""
    LOGGER.debug('%s Error: %s', tool_use_id, message)
    return models.Result(
        tool_use_id=tool_use_id, content=f'Error: {message}', is_error=True
    )


def success_result(tool_use_id: str, message: str) -> models.Result:
    """Set the success message and return the result as a dict."""
    return models.Result(tool_use_id=tool_use_id, content=message)

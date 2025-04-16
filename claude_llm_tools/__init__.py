from importlib import metadata

from .errors import ToolError, ValidationError
from .main import add_tool, dispatch, tool, tools
from .models import Request, Result

version = metadata.version('claude-llm-tools')

__all__ = [
    'Request',
    'Result',
    'ToolError',
    'ValidationError',
    'add_tool',
    'dispatch',
    'tool',
    'tools',
    'version',
]

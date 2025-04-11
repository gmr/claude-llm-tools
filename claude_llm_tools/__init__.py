from importlib import metadata

from .main import add_tool, dispatch, error_result, success_result, tool, tools
from .models import Request, Result

version = metadata.version('claude-llm-tools')

__all__ = [
    'Request',
    'Result',
    'add_tool',
    'dispatch',
    'error_result',
    'success_result',
    'tool',
    'tools',
    'version',
]

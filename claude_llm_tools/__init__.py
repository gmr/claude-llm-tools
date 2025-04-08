from importlib import metadata

from .main import add_tool, dispatch, error_result, success_result, tool, tools
from .models import Request, Result

try:
    version = metadata.version('claude-llm-tools')
except metadata.PackageNotFoundError:  # pragma: nocover
    version = '0.0.0.dev0'  # Default development version

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

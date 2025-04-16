"""Exceptions for claude-llm-tools"""


class ToolError(Exception):
    """Raised by a Tool when there is an error with the tool request"""

    pass


class ValidationError(Exception):
    """Raised when a tool request validation is not successful"""

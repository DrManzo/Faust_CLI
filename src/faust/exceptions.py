"""
Custom exception hierarchy for Faust.
"""


class FaustError(Exception):
    """Base exception for all Faust errors."""


class BackendError(FaustError):
    """Raised when an LLM backend fails to respond or is unreachable."""


class ConfigError(FaustError):
    """Raised when configuration is missing, malformed, or invalid."""


class ContextOverflowError(FaustError):
    """Raised when session context cannot be trimmed to fit the window."""


class GraphError(FaustError):
    """Raised when a LangGraph StateGraph fails to compile or execute."""

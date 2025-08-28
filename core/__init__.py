"""Core module for the web ingestion application."""

from .logging import configure_logging, get_logger, debug, info, warning, error, critical

__all__ = ["configure_logging", "get_logger", "debug", "info", "warning", "error", "critical"]
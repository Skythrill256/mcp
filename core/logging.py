from __future__ import annotations

import logging
import sys

from core.config.settings import AppSettings


def configure_logging(settings: AppSettings) -> None:
    """Configure logging for the application."""
    # see: https://gist.github.com/nymous/f138c7f06062b7c43c060bf03754c292
    # Import structlog lazily so importing this module doesn't fail when
    # structlog is not installed (e.g., in minimal dev envs).
    try:
        import structlog  # type: ignore
    except Exception:
        structlog = None  # type: ignore

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    
    logging.basicConfig(
        level=settings.log_level,
        handlers=handlers,
        format="%(message)s",
    )
    if structlog is not None:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    # Ensure no file-based handlers remain attached to the root logger.
    # This prevents accidental creation of log files by third-party libs or
    # imported modules that add FileHandlers earlier.
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        # Remove handlers that are file-based. Some handlers (e.g. FileHandler,
        # RotatingFileHandler) inherit from logging.FileHandler; others may
        # expose a `baseFilename` attribute depending on implementation.
        if isinstance(h, logging.FileHandler) or getattr(h, "baseFilename", None):
            root_logger.removeHandler(h)
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from core.config.settings import AppSettings

# Try to use loguru if available; otherwise fall back to stdlib logging
try:  # pragma: no cover - trivial import branch
    from loguru import logger as _loguru_logger  # type: ignore

    _HAS_LOGURU = True
except Exception:  # ImportError or other env issues
    _HAS_LOGURU = False
    _loguru_logger = None  # type: ignore
    import logging

    _stdlib_logger = logging.getLogger("pebbling-crawl")
    _stdlib_logger.propagate = False

# Expose a `logger` object either from loguru or stdlib
if _HAS_LOGURU:
    logger = _loguru_logger  # type: ignore[assignment]
else:
    logger = _stdlib_logger  # type: ignore[name-defined]


def configure_logging(settings: AppSettings) -> None:
    """Configure centralized logging using loguru when available, else stdlib."""
    if _HAS_LOGURU:
        # Remove default handlers
        logger.remove()

        # Configure log format (loguru syntax)
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        # Add stdout handler
        logger.add(
            sys.stdout,
            level=settings.log_level,
            format=log_format,
            enqueue=True,  # Thread-safe logging
            backtrace=True,
            diagnose=True,
        )

        # Add file handler if log_file is specified
        if settings.log_file:
            log_path = Path(settings.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                log_path,
                level=settings.log_level,
                format=log_format,
                enqueue=True,
                backtrace=True,
                diagnose=True,
                rotation="10 MB",  # Rotate after 10 MB
                retention="1 week",  # Keep logs for 1 week
                compression="zip",  # Compress rotated logs
            )

        # Intercept standard logging messages into loguru
        import logging

        class InterceptHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
                # Get corresponding Loguru level if possible
                level: str | int
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno

                # Find caller from where originated the logged message
                frame, depth = logging.currentframe(), 2
                while frame and frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back  # type: ignore
                    depth += 1

                logger.opt(depth=depth, exception=record.exc_info).log(
                    level, record.getMessage()
                )

        # Intercept standard logging
        logging.basicConfig(handlers=[InterceptHandler()], level=0)
    else:
        # Configure stdlib logging
        import logging

        # Clear existing handlers on our logger
        if logger.handlers:  # type: ignore[attr-defined]
            for h in list(logger.handlers):  # type: ignore[attr-defined]
                logger.removeHandler(h)  # type: ignore[attr-defined]

        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

        std_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        formatter = logging.Formatter(std_format, datefmt="%Y-%m-%d %H:%M:%S.%f")

        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        logger.addHandler(stream)  # type: ignore[attr-defined]

        if settings.log_file:
            log_path = Path(settings.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)  # type: ignore[attr-defined]


def get_logger(name: Optional[str] = None):
    """Get a logger instance (loguru or stdlib). Name is ignored for loguru."""
    if _HAS_LOGURU:
        return logger
    # For stdlib, return a child logger if name provided
    if name:
        import logging

        return logging.getLogger(name)
    return logger


# Convenience functions for different log levels
def debug(message: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message."""
    logger.debug(message, *args, **kwargs)


def info(message: str, *args: Any, **kwargs: Any) -> None:
    """Log an info message."""
    logger.info(message, *args, **kwargs)


def warning(message: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message."""
    logger.warning(message, *args, **kwargs)


def error(message: str, *args: Any, **kwargs: Any) -> None:
    """Log an error message."""
    logger.error(message, *args, **kwargs)


def critical(message: str, *args: Any, **kwargs: Any) -> None:
    """Log a critical message."""
    logger.critical(message, *args, **kwargs)

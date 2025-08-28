from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from core.config.settings import AppSettings


def configure_logging(settings: AppSettings) -> None:
    """Configure centralized logging for the application using loguru."""
    # Remove default handlers
    logger.remove()
    
    # Configure log format
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
        diagnose=True
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
            compression="zip"  # Compress rotated logs
        )
    
    # Intercept standard logging messages
    import logging
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Get corresponding Loguru level if possible
            level: str | int
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back  # type: ignore
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0)


def get_logger(name: str = None) -> "logger":
    """Get a logger instance with optional name."""
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
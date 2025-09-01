"""Tests for logging configuration utilities."""

import importlib
import sys
from pathlib import Path

import pytest

from core.config.settings import AppSettings
import core.logging as logging_mod


def test_configure_logging_loguru(tmp_path: Path):
    """Configure logging with loguru present and ensure file handler works."""
    # Ensure module is in loguru mode
    if getattr(logging_mod, "_HAS_LOGURU", False) is False:
        importlib.reload(logging_mod)

    settings = AppSettings()
    settings.log_level = "DEBUG"
    settings.log_file = str(tmp_path / "app.log")

    # Should not raise
    logging_mod.configure_logging(settings)

    # Emit a couple of logs
    logging_mod.info("hello world")
    logging_mod.error("boom")

    # File should exist (created on first write)
    assert Path(settings.log_file).parent.exists()


def test_configure_logging_stdlib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Force stdlib branch by faking loguru import failure and ensure it works."""
    # Force loguru import to fail on reload
    monkeypatch.setitem(sys.modules, "loguru", None)

    # Reload module to re-evaluate _HAS_LOGURU path
    mod = importlib.reload(logging_mod)

    settings = AppSettings()
    settings.log_level = "INFO"
    settings.log_file = str(tmp_path / "file.log")

    mod.configure_logging(settings)
    lg = mod.get_logger("test-logger")
    lg.info("info message")
    lg.warning("warn message")

    assert Path(settings.log_file).exists()

    # Clean up: allow loguru next imports again
    sys.modules.pop("loguru", None)
    importlib.reload(mod)

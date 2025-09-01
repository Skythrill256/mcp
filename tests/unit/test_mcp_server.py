"""Tests for MCP server helper."""

from unittest.mock import AsyncMock, patch

import pytest

from core.config.settings import AppSettings
from core.services.mcp_server import spawn_site_mcp


@pytest.mark.asyncio
async def test_spawn_site_mcp_minimal(monkeypatch: pytest.MonkeyPatch):
    cfg = AppSettings()
    cfg.mcp_port = None

    # Mock internal heavy deps
    # Provide a fake engine class in the target module's global namespace by importing it first
    import core.services.mcp_server as mcp_mod

    class FakeEngine:  # minimal stub
        def __init__(self, cfg):
            self.cfg = cfg

        async def query(self, **kwargs):  # pragma: no cover - not called here
            return {"answer": "ok"}

    with (
        patch.object(mcp_mod, "FastMCP") as m_fast,
        patch.object(mcp_mod, "LlamaIndexQueryEngine", FakeEngine),
        patch.object(mcp_mod, "find_free_port", return_value=12345),
        patch.object(mcp_mod, "asyncio") as m_asyncio,
        patch.object(mcp_mod, "discover_sitemaps", AsyncMock()),
    ):
        m_fast.return_value.tool = lambda f: f  # decorator passthrough
        m_asyncio.get_event_loop.return_value.create_task = lambda coro: None

        out = await spawn_site_mcp("https://example.com/x", cfg)
        assert out["port"] == 12345
        assert out["host"] == "127.0.0.1"
        assert out["path"].startswith("/mcp/")

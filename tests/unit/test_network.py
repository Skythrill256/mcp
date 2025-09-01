"""Tests for network helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.utils.network import find_free_port, discover_sitemaps


def test_find_free_port_preferred():
    assert find_free_port(9999) == 9999


def test_find_free_port_auto():
    port = find_free_port()
    assert isinstance(port, int) and 0 < port < 65536


@pytest.mark.asyncio
async def test_discover_sitemaps_success():
    # Mock httpx.AsyncClient.get to simulate sitemap and robots
    async def _aget(url, *_, **__):
        if url.endswith("/sitemap.xml"):
            return SimpleNamespace(status_code=200, text="<xml>ok</xml>")
        if url.endswith("/wp-sitemap.xml"):
            return SimpleNamespace(status_code=404, text="not found")
        if url.endswith("/robots.txt"):
            return SimpleNamespace(
                status_code=200, text="Sitemap: https://ex.com/sm.xml"
            )
        return SimpleNamespace(status_code=404, text="")

    mock_client = SimpleNamespace(get=AsyncMock(side_effect=_aget))

    class _AC:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, *args):
            return False

    with patch("core.utils.network.httpx.AsyncClient", _AC):
        urls = await discover_sitemaps("https://ex.com/path")
        # Should include explicit sitemap and one from robots
        assert "https://ex.com/sitemap.xml" in urls
        assert "https://ex.com/sm.xml" in urls


@pytest.mark.asyncio
async def test_discover_sitemaps_errors_are_swallowed():
    # Simulate network errors; function should handle and return [] (or minimal results)
    async def _aget(*a, **k):
        raise RuntimeError("net down")

    mock_client = SimpleNamespace(get=AsyncMock(side_effect=_aget))

    class _AC:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, *args):
            return False

    with patch("core.utils.network.httpx.AsyncClient", _AC):
        urls = await discover_sitemaps("https://ex.com")
        assert urls == []

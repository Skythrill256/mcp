from __future__ import annotations

import asyncio
import socket
from typing import Any, Optional
from urllib.parse import urlparse, urljoin

import httpx
from fastmcp import FastMCP, Context

from core.config.settings import AppSettings
from core.embeddings.embedding import EmbeddingManager


def _hostname_from_url(url: str) -> str:
    host = urlparse(url).netloc.replace(":", "_").replace(".", "_")
    return host or "site"


def _find_free_port(preferred: Optional[int] = None) -> int:
    if preferred:
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _discover_sitemaps(base_url: str) -> list[str]:
    discovered: list[str] = []
    origin = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        # common sitemap paths
        for sm_path in ("/sitemap.xml", "/wp-sitemap.xml"):
            sitemap_url = urljoin(origin, sm_path)
            try:
                resp = await client.get(sitemap_url)
                if resp.status_code == 200 and "<" in resp.text:
                    discovered.append(sitemap_url)
            except Exception:
                pass
        # robots.txt hints
        try:
            robots_url = urljoin(origin, "/robots.txt")
            resp = await client.get(robots_url)
            if resp.status_code == 200:
                for line in resp.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        sm = line.split(":", 1)[1].strip()
                        discovered.append(sm)
        except Exception:
            pass
    # dedupe
    return list(dict.fromkeys(discovered))


def spawn_site_mcp(site_url: str, base_config: AppSettings) -> dict[str, Any]:
    """Create and launch a FastMCP server for a given site.

    Returns dict with host, port, path, and server reference.
    """
    cfg = base_config.for_site(site_url)
    host_key = _hostname_from_url(site_url)

    mcp = FastMCP(name=f"SiteMCP::{host_key}")
    embed_mgr = EmbeddingManager(cfg)

    @mcp.tool
    async def ask(question: str, top_k: int = 5, ctx: Context | None = None) -> dict[str, Any]:
        """Ask questions grounded on the site's embedded content in Qdrant."""
        if ctx:
            await ctx.info(f"Querying vector store for: {question}")
        return await embed_mgr.query_llamaindex(question=question, top_k=top_k)

    @mcp.tool
    async def ask_metadata(ctx: Context | None = None) -> dict[str, Any]:
        """Return site metadata such as discovered sitemap.xml and robots.txt presence."""
        if ctx:
            await ctx.info("Discovering sitemaps and robots.txt")
        sitemaps = await _discover_sitemaps(site_url)
        meta = {"site": site_url, "sitemaps": sitemaps}
        # Try to fetch robots content briefly
        try:
            origin = f"{urlparse(site_url).scheme}://{urlparse(site_url).netloc}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(urljoin(origin, "/robots.txt"))
                meta["robots_txt"] = resp.text if resp.status_code == 200 else None
        except Exception:
            meta["robots_txt"] = None
        return meta

    port = _find_free_port(base_config.mcp_port)
    path = f"/mcp/{host_key}"

    # Run server in background thread using HTTP transport so we don't block the event loop
    async def _run_threaded():
        await asyncio.to_thread(
            mcp.run,  # blocking
            transport="http",
            host="127.0.0.1",
            port=port,
            path=path,
        )

    asyncio.get_event_loop().create_task(_run_threaded())

    return {
        "host": "127.0.0.1",
        "port": port,
        "path": path,
        "name": mcp.name,
    }

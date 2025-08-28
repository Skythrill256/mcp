from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlparse, urljoin

import httpx
from fastmcp import FastMCP, Context

from core.config.settings import AppSettings
# Query engine is imported lazily inside the function to avoid heavy startup imports
from core.utils.network import find_free_port, discover_sitemaps


def _hostname_from_url(url: str) -> str:
    host = urlparse(url).netloc.replace(":", "_").replace(".", "_")
    return host or "site"


async def spawn_site_mcp(site_url: str, base_config: AppSettings) -> dict[str, Any]:
    """Create and launch a FastMCP server for a given site.

    Returns dict with host, port, path, and server reference.
    """
    cfg = base_config.for_site(site_url)
    host_key = _hostname_from_url(site_url)

    mcp = FastMCP(name=f"SiteMCP::{host_key}")
    # Lazily import the query engine to avoid heavy imports at module import time
    from ..providers.llamaindex_query_engine import LlamaIndexQueryEngine
    query_engine = LlamaIndexQueryEngine(cfg)

    @mcp.tool
    async def ask(question: str, top_k: int = 5, use_reranking: bool = True, ctx: Context | None = None) -> dict[str, Any]:
        """Ask questions grounded on the site's embedded content in Qdrant."""
        if ctx:
            await ctx.info(f"Querying vector store for: {question}")
        return await query_engine.query(question=question, top_k=top_k, use_reranking=use_reranking)

    @mcp.tool
    async def ask_metadata(ctx: Context | None = None) -> dict[str, Any]:
        """Return site metadata such as discovered sitemap.xml and robots.txt presence."""
        if ctx:
            await ctx.info("Discovering sitemaps and robots.txt")
        sitemaps = await discover_sitemaps(site_url)
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

    port = find_free_port(base_config.mcp_port)
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

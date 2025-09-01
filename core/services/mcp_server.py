from __future__ import annotations

import asyncio
from typing import Any, Optional, Dict
from urllib.parse import urlparse, urljoin

import httpx
from fastmcp import FastMCP, Context

from core.config.settings import AppSettings
from core.utils.network import find_free_port, discover_sitemaps

# Expose a patchable reference for the query engine; import may be heavy so keep optional
try:  # pragma: no cover - import environment dependent
    from ..providers.llamaindex_query_engine import (
        LlamaIndexQueryEngine as _LlamaIndexQueryEngine,
    )
except Exception:  # pragma: no cover - optional dependency may be missing
    _LlamaIndexQueryEngine = None  # type: ignore

# Public alias that tests can patch: core.services.mcp_server.LlamaIndexQueryEngine
LlamaIndexQueryEngine = _LlamaIndexQueryEngine  # type: ignore


def _hostname_from_url(url: str) -> str:
    host = urlparse(url).netloc.replace(":", "_").replace(".", "_")
    return host or "site"


async def spawn_site_mcp(
    site_url: str, base_config: AppSettings, collection_name: Optional[str] = None
) -> dict[str, Any]:
    """Create and launch a FastMCP server for a given site.

    Returns dict with host, port, path, and server reference.
    """
    # Use the provided collection name if available, otherwise generate one
    cfg = base_config.for_site(site_url, collection_name=collection_name)
    host_key = _hostname_from_url(site_url)

    mcp_name = (
        f"SiteMCP::{collection_name}" if collection_name else f"SiteMCP::{host_key}"
    )

    mcp = FastMCP(name=mcp_name)
    # Use patchable alias if available, else import lazily
    engine_cls = LlamaIndexQueryEngine
    if engine_cls is None:  # type: ignore[truthy-bool]
        from ..providers.llamaindex_query_engine import (
            LlamaIndexQueryEngine as _LlamaIndexQueryEngine2,
        )

        engine_cls = _LlamaIndexQueryEngine2

    query_engine = engine_cls(cfg)  # type: ignore[misc]

    @mcp.tool
    async def ask(
        question: str,
        top_k: int = 5,
        use_reranking: bool = True,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Ask questions grounded on the site's embedded content in Qdrant."""
        if ctx:
            await ctx.info(f"Querying vector store for: {question}")
        return await query_engine.query(
            question=question, top_k=top_k, use_reranking=use_reranking
        )

    @mcp.tool
    async def ask_metadata(ctx: Context | None = None) -> dict[str, Any]:
        """Return site metadata such as discovered sitemap.xml and robots.txt presence."""
        if ctx:
            await ctx.info("Discovering sitemaps and robots.txt")
        sitemaps = await discover_sitemaps(site_url)
        meta: Dict[str, Any] = {"site": site_url, "sitemaps": sitemaps}
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

    loop = asyncio.get_event_loop()
    # In tests, loop may be patched; use create_task if available else schedule via ensure_future
    task_coro = _run_threaded()
    if hasattr(loop, "create_task"):
        loop.create_task(task_coro)
    else:  # pragma: no cover - fallback path
        asyncio.ensure_future(task_coro)

    return {
        "host": "127.0.0.1",
        "port": port,
        "path": path,
        "name": mcp.name,
    }

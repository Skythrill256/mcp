import socket
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx


def find_free_port(preferred: Optional[int] = None) -> int:
    if preferred:
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def discover_sitemaps(base_url: str) -> list[str]:
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

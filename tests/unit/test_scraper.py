"""Tests for the WebScraper service."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from core.services.scraper import WebScraper
from core.config.settings import AppSettings
from core.errors.exceptions import ScrapingError


@pytest.fixture
def mock_config():
    """Create a mock AppSettings for testing."""
    config = Mock(spec=AppSettings)
    config.keywords = ["test"]
    config.url_patterns = ["/test/*"]
    config.include_external = False
    config.max_pages = 5
    config.max_depth = 2
    config.collection_name = "test_collection"
    return config


@pytest.mark.asyncio
async def test_web_scraper_context_manager(mock_config):
    """Test that the WebScraper can be used as an async context manager."""
    with patch("core.services.scraper.AsyncWebCrawler") as mock_crawler_class:
        mock_crawler_instance = AsyncMock()
        mock_crawler_class.return_value = mock_crawler_instance

        async with WebScraper(mock_config) as scraper:
            assert scraper.crawler is not None

        # Verify that the crawler's context manager methods were called
        mock_crawler_instance.__aenter__.assert_called_once()
        mock_crawler_instance.__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_web_scraper_scrape_without_crawler():
    """Test that scraping without initializing the crawler raises an error."""
    config = AppSettings()
    scraper = WebScraper(config)

    with pytest.raises(
        ScrapingError, match="Use async context manager to initialize the scraper"
    ):
        await scraper.scrape_website("https://example.com")


def test_web_scraper_make_dispatcher(mock_config):
    """Test that the dispatcher is created correctly."""
    scraper = WebScraper(mock_config)
    dispatcher = scraper._make_dispatcher()

    # We can't easily test the actual type since it's from an external library
    # but we can verify it's created
    assert dispatcher is not None


def test_web_scraper_make_run_config(mock_config):
    """Test that the run config is created correctly."""
    scraper = WebScraper(mock_config)
    run_config = scraper._make_run_config()

    # Verify that the config has the expected attributes
    assert hasattr(run_config, "cache_mode")
    assert hasattr(run_config, "markdown_generator")


def test_web_scraper_create_crawl_strategy(mock_config):
    """Test that the crawl strategy is created correctly."""
    scraper = WebScraper(mock_config)
    strategy = scraper._create_crawl_strategy()

    # We can't easily test the actual type since it's from an external library
    # but we can verify it's created
    assert strategy is not None


def test_web_scraper_is_internal_link(mock_config):
    """Test the internal link detection."""
    scraper = WebScraper(mock_config)

    # Test internal links
    assert scraper._is_internal_link("https://example.com", "https://example.com/page")
    assert scraper._is_internal_link("https://example.com", "/page")
    assert scraper._is_internal_link("https://example.com", "page")

    # Test external links
    assert not scraper._is_internal_link(
        "https://example.com", "https://other.com/page"
    )

    # Test edge cases
    assert scraper._is_internal_link(
        "https://example.com", ""
    )  # Empty URL treated as internal

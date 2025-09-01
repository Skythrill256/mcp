"""Tests for database configuration utilities."""

from pathlib import Path
import yaml

from core.config.database import DatabaseConfig, load_db_settings


def test_database_config_load_and_pick(tmp_path: Path, monkeypatch):
    cfg_dir = tmp_path
    yaml_path = cfg_dir / "config-database.yaml"
    yaml_content = {
        "preferred_endpoint": "qdrant",
        "write_endpoint": "pgvector",
        "endpoints": {
            "qdrant": {
                "api_endpoint_env": "QDRANT_URL",
                "index_name": "web_vectors",
                "db_type": "qdrant",
                "enabled": True,
            },
            "pgvector": {
                "api_endpoint_env": "POSTGRES_CONNECTION_STRING",
                "index_name": "site_vectors",
                "db_type": "pgvector",
            },
        },
    }
    yaml_path.write_text(yaml.safe_dump(yaml_content))

    dc = DatabaseConfig(config_directory=str(cfg_dir))
    dc.load()
    name, ep = dc.pick_active_endpoint()
    # write_endpoint takes precedence if present
    assert name == "pgvector"
    assert ep.index_name == "site_vectors"

    # Ensure helper returns DBSettings shape
    settings = load_db_settings(str(yaml_path))
    assert settings.db_type in ("qdrant", "pgvector")
    assert isinstance(settings.index_name, str)


def test_database_config_default_when_missing(tmp_path: Path):
    # No file -> falls back to defaults
    dc = DatabaseConfig(config_directory=str(tmp_path))
    dc.load("nonexistent.yaml")
    name, ep = dc.pick_active_endpoint()
    assert name in ("qdrant", "pgvector", "postgres")

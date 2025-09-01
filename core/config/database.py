from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import yaml  # type: ignore[import-untyped]


def _default_config_path() -> str:
    # default to this module directory (core/config)
    return os.path.join(os.path.dirname(__file__), "config-database.yaml")


@dataclass
class RetrievalProviderConfig:
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_endpoint_env: Optional[str] = None
    database_path: Optional[str] = None
    index_name: Optional[str] = None
    db_type: Optional[str] = None
    enabled: bool = False
    use_knn: Optional[bool] = None
    vector_type: Optional[str] = None


@dataclass
class DBSettings:
    db_type: str
    api_endpoint_env: Optional[str]
    connection: Optional[str]
    index_name: str
    name: str


class DatabaseConfig:
    def __init__(self, config_directory: Optional[str] = None):
        self.config_directory = config_directory or os.path.dirname(__file__)
        self.retrieval_endpoints: Dict[str, RetrievalProviderConfig] = {}
        self.write_endpoint: Optional[str] = None
        self.preferred_endpoint: Optional[str] = None

    def load(self, path: str = "config-database.yaml") -> None:
        full_path = os.path.join(self.config_directory, path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: {path} not found. Using default database configuration.")
            data = {
                "preferred_endpoint": "qdrant",
                "endpoints": {},
            }

        # legacy field support
        self.preferred_endpoint = data.get("preferred_endpoint")
        self.write_endpoint = data.get("write_endpoint")

        # Build endpoints
        self.retrieval_endpoints.clear()
        for name, cfg in (data.get("endpoints") or {}).items():
            self.retrieval_endpoints[name] = RetrievalProviderConfig(
                api_key=_get_env_or_value(cfg.get("api_key_env")),
                api_key_env=cfg.get("api_key_env"),
                api_endpoint=_get_env_or_value(cfg.get("api_endpoint_env")),
                api_endpoint_env=cfg.get("api_endpoint_env"),
                database_path=cfg.get("database_path"),
                index_name=cfg.get("index_name"),
                db_type=(cfg.get("db_type") or name),
                enabled=cfg.get("enabled", False),
                use_knn=cfg.get("use_knn"),
                vector_type=cfg.get("vector_type"),
            )

        # If no explicit enabled flags, enable the preferred one for backward compat
        if not any(ep.enabled for ep in self.retrieval_endpoints.values()):
            if (
                self.preferred_endpoint
                and self.preferred_endpoint in self.retrieval_endpoints
            ):
                self.retrieval_endpoints[self.preferred_endpoint].enabled = True

    def pick_active_endpoint(self) -> Tuple[str, RetrievalProviderConfig]:
        # Priority: write_endpoint -> any enabled -> preferred -> first available
        if self.write_endpoint and self.write_endpoint in self.retrieval_endpoints:
            name = self.write_endpoint
            return name, self.retrieval_endpoints[name]
        for name, cfg in self.retrieval_endpoints.items():
            if cfg.enabled:
                return name, cfg
        if (
            self.preferred_endpoint
            and self.preferred_endpoint in self.retrieval_endpoints
        ):
            name = self.preferred_endpoint
            return name, self.retrieval_endpoints[name]
        # fallback to first
        if self.retrieval_endpoints:
            name = next(iter(self.retrieval_endpoints.keys()))
            return name, self.retrieval_endpoints[name]
        # ultimate default to qdrant
        return "qdrant", RetrievalProviderConfig(
            api_endpoint=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_endpoint_env="QDRANT_URL",
            index_name="web_vectors",
            db_type="qdrant",
            enabled=True,
        )


def _get_env_or_value(env_name: Optional[str]) -> Optional[str]:
    if not env_name:
        return None
    return os.getenv(env_name)


def load_db_settings(path: Optional[str] = None) -> DBSettings:
    """Backward-compatible helper returning a single active endpoint config."""
    cfg = DatabaseConfig()
    cfg.load(path or "config-database.yaml")
    name, ep = cfg.pick_active_endpoint()

    # For compatibility: determine connection string and env var name
    env_var = ep.api_endpoint_env or (
        "QDRANT_URL"
        if (ep.db_type or name).lower() == "qdrant"
        else "POSTGRES_CONNECTION_STRING"
    )
    connection = ep.api_endpoint or os.getenv(env_var) if env_var else None
    index = ep.index_name or "web_vectors"
    db_type = (ep.db_type or name or "qdrant").lower()
    if db_type == "postgres":
        db_type = "pgvector"

    return DBSettings(
        db_type=db_type,
        api_endpoint_env=env_var,
        connection=connection,
        index_name=index,
        name=name,
    )

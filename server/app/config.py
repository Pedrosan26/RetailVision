"""
config.py

Server configuration, loaded from environment variables (or a .env file
in development). Centralizes settings so nothing else in the app reads
os.environ directly, and exposes get_settings() so routes/dependencies
can override it in tests instead of depending on process-wide state.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server configuration loaded from the environment, with local-dev-friendly defaults."""

    database_url: str = "postgresql+asyncpg://retailvision:retailvision@localhost:5432/retailvision"
    camera_node_api_keys: str = ""
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env")

    def camera_node_api_key_map(self) -> dict[str, str]:
        """Parse CAMERA_NODE_API_KEYS ("node1:key1,node2:key2") into a {node_id: key} dict."""
        pairs = (pair.split(":", 1) for pair in self.camera_node_api_keys.split(",") if pair)
        return {node_id.strip(): key.strip() for node_id, key in pairs}

    def cors_origin_list(self) -> list[str]:
        """Parse CORS_ORIGINS into a list of allowed origins."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings instance, loaded once from the environment/.env file."""
    return Settings()

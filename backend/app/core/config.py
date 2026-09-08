"""Application settings, loaded from environment with sane local defaults."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SATQUERY_", extra="ignore"
    )

    app_name: str = "SatQuery AI"
    api_prefix: str = "/api"

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "satquery"
    # Kept short so a missing Mongo never stalls startup or /health.
    mongo_timeout_ms: int = 3000

    model_config_path: Path = BACKEND_ROOT / "model_config.yaml"
    upload_dir: Path = BACKEND_ROOT / "storage" / "uploads"

    max_upload_bytes: int = 512 * 1024 * 1024  # 512 MB, GeoTIFFs get large
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    # Intent classification. OpenRouter fronts both DeepSeek and Gemini, so the
    # provider changes with this model id alone. Without a key the graph falls
    # back to keyword classification and says so in the trace.
    openrouter_api_key: str | None = None
    openrouter_model: str = "deepseek/deepseek-chat"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_referer: str | None = None
    openrouter_timeout_s: float = 20.0


@lru_cache
def get_settings() -> Settings:
    return Settings()

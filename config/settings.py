"""
Centralized configuration management for the Health Assistant.
Uses pydantic-settings for environment variable validation and type safety.
"""

from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    google_api_key: str = Field(..., description="Google API key for Gemini")
    groq_api_key: str = Field(..., description="Groq API key for Whisper")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="healthassistant123")

    # Redis Configuration
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)

    # Application Settings
    app_name: str = Field(default="Health Assistant")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    default_language: Literal["en", "ru"] = Field(default="en")

    # Rate Limiting
    gemini_max_requests_per_minute: int = Field(default=15)
    gemini_max_requests_per_day: int = Field(default=1500)
    groq_whisper_max_requests_per_minute: int = Field(default=30)
    groq_whisper_max_requests_per_day: int = Field(default=14400)
    tts_monthly_char_limit: int = Field(default=1_000_000)

    # Voice Settings
    groq_whisper_model: str = Field(default="whisper-large-v3")
    tts_voice_en: str = Field(default="en-US-Neural2-J")
    tts_voice_ru: str = Field(default="ru-RU-Wavenet-D")

    # Caching
    cache_ttl_seconds: int = Field(default=3600)
    semantic_similarity_threshold: float = Field(default=0.85)

    # ChromaDB
    chromadb_persist_directory: Path = Field(default=Path("./data/chromadb"))
    chromadb_collection_name: str = Field(default="health_memories")

    # Computed Properties
    @property
    def daily_tts_char_limit(self) -> int:
        """Calculate daily TTS character limit from monthly limit"""
        return self.tts_monthly_char_limit // 30

    @field_validator("chromadb_persist_directory")
    def create_chromadb_directory(cls, v: Path) -> Path:
        """Ensure ChromaDB directory exists"""
        v.mkdir(parents=True, exist_ok=True)
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
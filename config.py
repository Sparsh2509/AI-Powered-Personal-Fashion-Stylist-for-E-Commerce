"""
config.py — Centralized configuration using environment variables.

WHY: Having one config file means you change your API key or model
name in ONE place (.env) and it automatically applies everywhere.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Reads all settings from the .env file automatically.
    Pydantic validates types so you catch missing keys early.
    """

    # Google Gemini API Key (from aistudio.google.com)
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")

    # ChromaDB local storage folder
    chroma_db_path: str = Field(default="./chroma_db", env="CHROMA_DB_PATH")

    # ChromaDB collection (like a "table" in a regular DB)
    chroma_collection_name: str = Field(
        default="fashion_knowledge", env="CHROMA_COLLECTION_NAME"
    )

    # Gemini model for analyzing photos (vision-capable)
    # For new SDK: model names don't need "models/" prefix
    gemini_vision_model: str = Field(
        default="gemini-2.5-flash-lite", env="GEMINI_VISION_MODEL"
    )

    # Gemini model for generating final text recommendations
    gemini_llm_model: str = Field(
        default="gemini-2.5-flash-lite", env="GEMINI_LLM_MODEL"
    )

    # Gemini model for creating embeddings (vector representations)
    gemini_embedding_model: str = Field(
        default="models/gemini-embedding-001", env="GEMINI_EMBEDDING_MODEL"
    )

    # FastAPI server settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    class Config:
        env_file = ".env"          # Load from .env file
        env_file_encoding = "utf-8"
        extra = "ignore"           # Ignore unknown env vars


# ---------------------------------------------------------------
# Single shared instance — import this everywhere
# ---------------------------------------------------------------
settings = Settings()

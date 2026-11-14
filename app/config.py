
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_name: str = "RAG Eval"

    # LLM configuration
    groq_model_name: str = "openai/gpt-oss-120b"

    # Embedding configuration
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"

    # API keys
    groq_api_key: str
    tavily_api_key: str
    pinecone_api_key: str

    # Pinecone configuration
    pinecone_index_name: str = "rag-eval-fde"
    pinecone_namespace: str = "rag-eval"

    # Retrieval configuration
    top_k: int = 4
    max_retries: int = 1

    # Security
    admin_api_key: str

    # Application paths
    audit_db_path: str = str(BASE_DIR / "audit_logs" / "audit.db")
    upload_dir: Path = BASE_DIR / "uploads"
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")
    model_path: Path = BASE_DIR / "models"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent[2]

class Settings(BaseSettings):
    app_name : str = "RAG Eval"
    groq_model_name: str = "openai/gpt-oss-120b"
    model_name: str = "gpt-oss-120b"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    groq_api_key: str
    tavily_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "rag-eval-FDE"
    pinecone_namespace: str = "rag-eval"
    top_k: int = 4
    max_retries: int = 1
    admin_api_key: str = "admin"
    audit_db_path: str = str(BASE_DIR / "audit_logs" / "audit.db")
    upload_dir: Path = BASE_DIR / "uploads"
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb") 

    model_path: Path = BASE_DIR / "models"

    model_path.mkdir(parents=True, exist_ok=True)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8") 
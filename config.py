"""Centralised settings & hyper-parameters."""
from pathlib import Path
from pydantic import Field
from typing import ClassVar, Dict
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_ENDPOINT: str = Field(..., env="OPENAI_ENDPOINT")
    PGVECTOR_URL: str = Field(..., env="PGVECTOR_URL")  # postgresql://user:pass@host/db
    AZURE_SEARCH_ENDPOINT: str = Field(..., env="AZURE_SEARCH_ENDPOINT")
    AZURE_SEARCH_KEY:      str = Field(..., env="AZURE_SEARCH_KEY")
    COLLECTION: str = "autorag_documents"
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 128
    QA_PER_CHUNK: int = 2
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    LLM_MODEL: str = "gpt-4o-mini"
    AUTORAG_METRIC: str = "context_precision"  # from RAGAS
    METADATA_COLUMNS: ClassVar[Dict[str, str]] = {"source": "text", "page": "int", "source_text": "jsonb"}
    class Config:
        env_file = Path(__file__).with_suffix(".env")

settings = Settings()
"""
Configuration — Healthcare AI Diagnostic Assistant
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production-xyz")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./db/healthcare.db")

    # Vector store
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "db/faiss_index")
    FAISS_METADATA_PATH: str = os.getenv("FAISS_METADATA_PATH", "db/faiss_metadata.json")

    # HIPAA
    AUDIT_LOG_PATH: str = os.getenv("AUDIT_LOG_PATH", "outputs/audit.log")
    PHI_MASKING_ENABLED: bool = os.getenv("PHI_MASKING_ENABLED", "true").lower() == "true"

    # FHIR
    FHIR_SERVER_URL: str = os.getenv("FHIR_SERVER_URL", "http://localhost:8080/fhir")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Clinical RAG
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.7
    MAX_CONTEXT_TOKENS: int = 6000


settings = Settings()

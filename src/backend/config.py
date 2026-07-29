"""Configuration settings for DocAssist."""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


# Project root directory resolution
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _load_env_files() -> None:
    """
    Load environment variables from common .env locations without overriding
    existing environment values.
    """
    candidates = [
        os.path.join(_ROOT_DIR, ".env"),
        os.path.join(_ROOT_DIR, "src", ".env"),
        os.path.join(_ROOT_DIR, "src", "backend", ".env"),
    ]
    for p in candidates:
        try:
            if os.path.isfile(p):
                load_dotenv(p, override=True)
        except Exception:
            # Best-effort loading; ignore any parsing issues to avoid crashing
            pass


class Settings(BaseSettings):
    # API Keys
    groq_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    cerebras_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None

    # Database
    database_url: str = f"sqlite:///{os.path.join(_ROOT_DIR, 'agentic_rag.db')}"

    # Vector Store
    # Supported: "pinecone" (primary), "faiss" (fallback), "pgvector" (future)
    vector_store_type: str = "pinecone"
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384

    # Pinecone (serverless) configuration
    pinecone_index_name: str = "agentic-rag-index"
    pinecone_cloud: str = "aws"  # aws|gcp|azure
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "cosine"  # cosine|dotproduct|euclidean
    pinecone_namespace: Optional[str] = None  # Optional logical namespace

    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_docs: int = 8

    # LLM Settings
    # Provider/model selection lives in src/backend/llm_factory.py.
    llm_model: str = "openai/gpt-oss-120b"
    groq_model: str = "openai/gpt-oss-120b"
    ollama_model: Optional[str] = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    nvidia_nim_base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048

    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = [".pdf", ".docx", ".txt", ".md"]
    upload_dir: str = os.path.join(_ROOT_DIR, "uploads")

    # Workflow Settings
    max_workflow_iterations: int = 3
    compliance_threshold: float = 0.7
    quality_threshold: float = 0.8

    # Feedback Settings
    min_feedback_score: int = 1  # Minimum feedback score (1-5 scale)
    max_feedback_score: int = 5  # Maximum feedback score (1-5 scale)

    # Agent Settings
    enable_mock_agents: bool = True  # Set to False when API keys are available

    class Config:
        env_file = ".env"
        case_sensitive = False


# Load .env files before instantiating Settings so pydantic sees them
_load_env_files()
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.upload_dir, exist_ok=True)


# Validate configuration
def validate_config():
    """Validate configuration and warn about missing API keys"""
    warnings = []

    if not settings.pinecone_api_key:
        warnings.append("Pinecone API key missing - falling back to FAISS")
    else:
        # Provide informational hints if user forgot to set optional fields
        if not settings.pinecone_index_name:
            warnings.append("Pinecone index name not set - using default 'agentic-rag-index'")
        if not settings.pinecone_region or not settings.pinecone_cloud:
            warnings.append("Pinecone region/cloud not fully set - defaulting to aws/us-east-1")

    if warnings:
        import sys
        print("Configuration warnings:", file=sys.stderr)
        for warning in warnings:
            print(f"   - {warning}", file=sys.stderr)
        print(file=sys.stderr)


# Run validation on import
validate_config()

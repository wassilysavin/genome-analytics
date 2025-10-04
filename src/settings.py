import logging
from typing import Optional
from pydantic_settings import BaseSettings

from .constants.constants import *

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # API Keys and Credentials
    xai_api_key: str = ""
    pinecone_api_key: Optional[str] = None
    ncbi_email: str = ""
    ncbi_api_key: Optional[str] = None

    # Model Configuration
    model_name: str = "grok-3-mini"
    temperature: float = 0.0
    embedding_model: str = "all-MiniLM-L6-v2"

    # Analysis Limits and Thresholds
    max_annotations: int = 3
    max_sequence_length: int = 1000000  # 1MB limit for sequences
    min_orf_length: int = 300
    max_genes_for_display: int = 100
    max_genes_for_summary: int = 100

    # RAG and Vector Store Configuration
    vector_store_type: str = "pinecone"
    collection_name: str = "genome-insights"
    chunk_size: int = 8000
    chunk_overlap: int = 1000
    retriever_k: int = 3
    vector_search_top_k: int = 5
    chat_vector_search_top_k: int = 3

    # Performance and Timeout Settings
    api_delay: float = 0.1
    request_timeout: int = 30
    blast_num_threads: int = 4
    search_default_max_results: int = 10
    rate_limit_delay_default: float = 0.1
    
    # Annotation Service Settings
    annotation_requests_per_second: float = 2.0
    annotation_max_concurrent_requests: int = 5
    annotation_max_retries: int = 3
    annotation_backoff_factor: float = 0.3
    annotation_connection_timeout: int = 10
    
    # BLAST and Prompt Settings
    blast_remote_default_max_results: int = 19
    prompt_default_max_response_length: int = 500
    default_top_genes_limit: int = 10
    default_strict_mode: bool = False
    prompt_default_concise_mode: bool = True

    log_level: str = "INFO"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.xai_api_key:
            raise ValueError("XAI_API_KEY is required. Please set it in your .env file.")

        if self.vector_store_type == "pinecone" and not self.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY is required when using Pinecone vector store. Please set it in your .env file or change vector_store_type."
            )

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "allow",  
    }


settings = Settings()

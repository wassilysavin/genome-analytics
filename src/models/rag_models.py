from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..constants.constants import *
from ..settings import settings


class FileType(Enum):
    FASTA = "fasta"
    FASTQ = "fastq"
    VCF = "vcf"
    AUTO = "auto"


class DocumentType(Enum):
    SEQUENCE = "sequence"
    VARIANT = "variant"


@dataclass
class VectorStoreConfig:
    collection_name: str
    dimension: int
    pinecone_api_key: str


@dataclass
class RAGConfig:
    embedding_model: str
    annotation_requests_per_second: float = DEFAULT_ANNOTATION_REQUESTS_PER_SECOND
    max_concurrent_requests: int = DEFAULT_ANNOTATION_MAX_CONCURRENT_REQUESTS
    annotation_max_retries: int = DEFAULT_ANNOTATION_MAX_RETRIES
    enable_annotation_cache: bool = True
    annotation_cache_ttl: int = ANNOTATION_CACHE_TTL
    collection_name: str = settings.collection_name
    dimension: int = RAG_DEFAULT_DIMENSION
    ncbi_api_key: Optional[str] = None
    omim_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None


@dataclass
class ProcessedDocument:
    content: str
    metadata: dict[str, Any]


@dataclass
class SearchResult:
    rank: int
    content: str
    metadata: dict[str, Any]
    similarity: float


@dataclass
class GenomeProcessResult:
    success: bool
    file_type: Optional[str] = None
    parsed_count: int = 0
    annotated_count: int = 0
    documents_count: int = 0
    data: Optional[list[dict[str, Any]]] = None
    documents: Optional[list[ProcessedDocument]] = None
    error: Optional[str] = None


@dataclass
class IndexBuildResult:
    success: bool
    documents_count: int = 0
    vector_store_stats: Optional[dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class RagChatResult:
    has_context: bool
    text: str
    top_similarity: Optional[float] = None

from dataclasses import dataclass
from typing import Any, Optional

from ..constants.constants import *


@dataclass
class AnnotationConfig:
    requests_per_second: float = DEFAULT_ANNOTATION_REQUESTS_PER_SECOND
    max_concurrent_requests: int = DEFAULT_ANNOTATION_MAX_CONCURRENT_REQUESTS

    max_retries: int = DEFAULT_ANNOTATION_MAX_RETRIES
    backoff_factor: float = DEFAULT_ANNOTATION_BACKOFF_FACTOR

    request_timeout: int = DEFAULT_ANNOTATION_REQUEST_TIMEOUT
    connection_timeout: int = DEFAULT_ANNOTATION_CONNECTION_TIMEOUT

    enable_cache: bool = True
    cache_ttl: int = ANNOTATION_CACHE_TTL

    ncbi_api_key: Optional[str] = None
    omim_api_key: Optional[str] = None


@dataclass
class VariantInfo:
    chrom: str
    pos: int
    ref: str
    alt: str
    rs_id: Optional[str] = None
    variant_id: Optional[str] = None


@dataclass
class AnnotationResult:
    status: str
    source: str
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    queried_at: float = 0.0


@dataclass
class VariantAnnotation:
    variant: VariantInfo
    sources: dict[str, AnnotationResult]
    annotated_at: float


@dataclass
class AnnotationSummary:
    total_variants: int
    successful_annotations: int
    success_rate: float
    source_statistics: dict[str, dict[str, int]]
    generated_at: float

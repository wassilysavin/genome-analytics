from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ComponentStatus(Enum):
    INITIALIZED = "initialized"
    NOT_INITIALIZED = "not_initialized"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass
class AnalysisResult:
    success: bool
    sequence_length: int
    results: dict[str, Any]
    method: str
    message: str
    error: Optional[str] = None


@dataclass
class AppGenomeProcessResult:
    success: bool
    file_type: Optional[str] = None
    parsed_count: int = 0
    annotated_count: int = 0
    documents_count: int = 0
    index_built: bool = False
    index_error: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Document:
    content: str
    metadata: dict[str, Any]


@dataclass
class ComponentStatusInfo:
    agent_type: str
    status: ComponentStatus
    details: Optional[dict[str, Any]] = None


@dataclass
class WorkflowStatus:
    traditional_gene_agent: Optional[ComponentStatusInfo] = None
    genome_rag: Optional[ComponentStatusInfo] = None

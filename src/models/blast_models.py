from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    VERY_LOW = "Very Low"


@dataclass
class BlastHit:
    hit_id: str
    gene_name: str
    description: str
    e_value: float
    identity: float
    length: int
    database: str
    confidence: str
    bitscore: Optional[float] = None
    query_start: Optional[int] = None
    query_end: Optional[int] = None
    subject_start: Optional[int] = None
    subject_end: Optional[int] = None


@dataclass
class BlastResult:
    found_matches: bool
    hits: list[BlastHit]
    total_results: int
    search_method: str
    error: Optional[str] = None


@dataclass
class DatabaseInfo:
    name: str
    path: str
    size_mb: float
    is_available: bool

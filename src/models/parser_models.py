from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SequenceRecord:
    id: str
    description: str
    sequence: str
    length: int
    name: Optional[str] = None
    quality: Optional[list[int]] = None


@dataclass
class VariantRecord:
    chrom: str
    pos: int
    id: Optional[str]
    ref: str
    alt: list[str]
    qual: Optional[float]
    info_keys: list[str]


@dataclass
class ParseResult:
    success: bool
    records: list[Any]
    count: int
    error: Optional[str] = None


@dataclass
class FastaParseResult:
    success: bool
    records: list[Any]
    count: int
    sequences: list[SequenceRecord]
    error: Optional[str] = None


@dataclass
class FastqParseResult:
    success: bool
    records: list[Any]
    count: int
    sequences: list[SequenceRecord]
    error: Optional[str] = None


@dataclass
class VcfParseResult:
    success: bool
    records: list[Any]
    count: int
    variants: list[VariantRecord]
    error: Optional[str] = None

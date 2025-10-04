from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AnalyzedORF:
    orf_id: str
    genomic_position: dict[str, Any]
    length_bp: int
    protein_sequence: str
    protein_properties: Optional[dict[str, Any]] = None
    frame: Optional[int] = None


@dataclass
class SequenceAnalysis:
    success: bool
    sequence_composition: dict[str, Any]
    total_orfs_found: int
    analyzed_orfs: list[AnalyzedORF]
    analysis_method: str
    error: Optional[str] = None


@dataclass
class DatabaseSearchResults:
    ncbi_protein_results: list[dict[str, Any]]
    ncbi_nucleotide_results: list[dict[str, Any]]
    uniprot_results: list[dict[str, Any]]
    total_matches: int = 0


@dataclass
class GeneIdentificationResult:
    success: bool
    sequence_length: int
    analysis_method: str
    sequence_analysis: Optional[SequenceAnalysis] = None
    database_search: Optional[DatabaseSearchResults] = None
    ai_summary: Optional[dict[str, Any]] = None
    detailed_insights: Optional[dict[str, Any]] = None
    total_database_matches: int = 0
    message: Optional[str] = None
    error: Optional[str] = None

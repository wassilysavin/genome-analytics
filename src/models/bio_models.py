from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class ORF:
    start: int
    end: int
    length: int
    strand: str
    frame: int
    sequence: str
    start_codon: str
    protein_sequence: Optional[str] = None

    @property
    def orf_id(self) -> str:
        return f"ORF_{self.start}_{self.end}_{self.strand}"


@dataclass
class ProteinProperties:
    length: int
    molecular_weight: float
    isoelectric_point: float
    instability_index: float
    aromaticity: float
    amino_acid_composition: dict[str, float]
    gravy: float
    secondary_structure: dict[str, float]
    stability: str
    hydrophobicity: str


@dataclass
class SequenceComposition:
    length: int
    gc_content: float
    nucleotide_counts: dict[str, int]
    nucleotide_percentages: dict[str, float]
    gc_assessment: str


@dataclass
class AnalysisError:
    error_type: str
    message: str
    details: Optional[str] = None

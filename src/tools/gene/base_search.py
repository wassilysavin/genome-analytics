import logging
import re
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional
from ...constants.constants import *
from ...settings import settings

logger = logging.getLogger(__name__)


class SearchType(Enum):
    PROTEIN = "protein"
    NUCLEOTIDE = "nucleotide"
    GENE_NAME = "gene_name"


class DatabaseType(Enum):
    NCBI = "ncbi"
    UNIPROT = "uniprot"
    BLAST_LOCAL = "blast_local"
    BLAST_REMOTE = "blast_remote"


class SearchResult:
    def __init__(
        self,
        found_matches: bool,
        hits: list[dict[str, Any]],
        total_results: int = 0,
        search_method: str = "",
        error: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.found_matches = found_matches
        self.hits = hits
        self.total_results = total_results
        self.search_method = search_method
        self.error = error
        self.database = database

    def to_dict(self) -> dict[str, Any]:
        return {
            "found_matches": self.found_matches,
            "hits": self.hits,
            "total_results": self.total_results,
            "search_method": self.search_method,
            "error": self.error,
            "database": self.database,
        }

    def is_successful(self) -> bool:
        return self.found_matches and self.error is None


class BaseSearchProvider(ABC):
    def __init__(self, database_type: DatabaseType, search_type: SearchType):
        self.database_type = database_type
        self.search_type = search_type
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def search(self, query: str, max_results: int = settings.search_default_max_results) -> SearchResult:
        pass

    @abstractmethod
    def setup(self) -> bool:
        pass

    def validate_query(self, query: str) -> bool:
        if not query or not isinstance(query, str):
            return False
        return not len(query.strip()) < SEARCH_MIN_QUERY_LENGTH

    def create_error_result(self, error_message: str) -> SearchResult:
        return SearchResult(
            found_matches=False,
            hits=[],
            error=error_message,
            search_method=f"{self.database_type.value} {self.search_type.value} search (failed)",
        )

    def rate_limit(self, delay: float = settings.rate_limit_delay_default):
        time.sleep(delay)


class SearchConfig:
    def __init__(self, **kwargs):
        self.max_results = kwargs.get("max_results", settings.search_default_max_results)
        self.sequence_length = kwargs.get("sequence_length", SEARCH_DEFAULT_SEQUENCE_LENGTH)
        self.nucleotide_length = kwargs.get("nucleotide_length", SEARCH_DEFAULT_NUCLEOTIDE_LENGTH)
        self.rate_limit_delay = kwargs.get("rate_limit_delay", settings.rate_limit_delay_default)
        self.blast_evalue_threshold = kwargs.get(
            "blast_evalue_threshold", BLAST_EXPECT_THRESHOLD
        )
        self.blast_gap_costs = kwargs.get("blast_gap_costs", BLAST_GAP_COSTS)
        self.blast_protein_word_size = kwargs.get(
            "blast_protein_word_size", BLAST_PROTEIN_WORD_SIZE
        )
        self.blast_nucleotide_word_size = kwargs.get(
            "blast_nucleotide_word_size", BLAST_NUCLEOTIDE_WORD_SIZE
        )


class SearchUtilities:
    @staticmethod
    def extract_gene_name(title: str) -> str:
        import re

        if not title or not isinstance(title, str):
            return UNKNOWN_LABEL

        # Common patterns for gene names
        patterns = [
            r"GN=([A-Za-z0-9_]+)",  # NCBI format: GN=INS
            r"\[gene=(\w+)\]",  # BLAST format: [gene=EGFR]
            r"gene=(\w+)",  # BLAST format: gene=BRCA1
            r"Gene=(\w+)",  # BLAST format: Gene=TP53
            r"\((\w+)\)",  # Parentheses: (ACTB)
            r"(\w+)\s+gene",  # "BRCA1 gene"
            r"(\w+)\s+protein",  # "TP53 protein"
            r"RecName:\s*Full=([^;\[\]]+)",  # Swiss-Prot format
            r"Short=([^;\[\]]+)",  # Swiss-Prot format
        ]

        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                gene_name = match.group(SEARCH_GENE_NAME_GROUP_INDEX).strip()
                if SearchUtilities._is_valid_gene_name(gene_name):
                    return gene_name

        # Trying extracting from pipe-separated format (NCBI format)
        pipe_parts = title.split("|")
        if len(pipe_parts) >= SEARCH_PIPE_PARTS_MIN:
            gene_part = pipe_parts[SEARCH_PIPE_GENE_PART_INDEX]
            if "_" in gene_part:
                gene_name = gene_part.split("_")[SEARCH_UNDERSCORE_SPLIT_INDEX]
                if SearchUtilities._is_valid_gene_name(gene_name):
                    return gene_name

        # Trying capitalized words
        words: list[str] = re.findall(r"\b[A-Z][A-Za-z0-9]{" + str(SEARCH_GENE_WORD_MIN_LENGTH) + ",}\b", title)
        for word in words:
            if SearchUtilities._is_valid_gene_name(word):
                return word

        return UNKNOWN_LABEL

    @staticmethod
    def _is_valid_gene_name(name: str) -> bool:
        if not name or len(name) < SEARCH_MIN_NAME_LENGTH:
            return False

        # Generic terms to filter out
        generic_terms = {
            "unknown",
            "protein",
            "putative",
            "uncharacterized",
            "pro",
            "communis",
            "filament-binding",
            "murc",
            "ddl",
            "gvqw1",
            "subunit",
            "cadherin-20",
            "c16orf89",
            "binding",
            "transporter",
            "synthase",
            "ligase",
            "transferase",
            "reductase",
            "oxidase",
            "dehydrogenase",
            "complex",
            "assembly",
            "precursor",
            "flags",
            "recname",
            "short",
            "full",
            "os",
            "pe",
            "sv",
            "tr",
            "sp",
            "mrna",
            "cdna",
            "partial",
            "complete",
        }

        # Filter out protein IDs (e.g., Q5NVH5.2, P02768.2)
        if re.match(r"^[A-Z]\d[A-Z0-9]{" + str(GENE_EXTRACTOR_PROTEIN_ID_MIN_LENGTH) + "," + str(GENE_EXTRACTOR_PROTEIN_ID_MAX_LENGTH) + "}(\.\d+)?$", name):
            return False

        if name.lower() in generic_terms:
            return False

        # Filter out names that are mostly numbers
        return not (sum(c.isdigit() for c in name) > len(name) * GENE_EXTRACTOR_DIGIT_THRESHOLD)

    @staticmethod
    def calculate_confidence(e_value: float) -> str:
        if e_value < SEARCH_CONFIDENCE_HIGH:
            return "Very High"
        elif e_value < SEARCH_CONFIDENCE_MEDIUM:
            return "High"
        elif e_value < SEARCH_CONFIDENCE_LOW:
            return "Medium"
        else:
            return "Low"

    @staticmethod
    def create_standardized_hit(
        hit_id: str,
        description: str,
        gene_name: Optional[str] = None,
        organism: Optional[str] = None,
        length: int = 0,
        score: Optional[float] = None,
        e_value: Optional[float] = None,
        database: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        hit: dict[str, Any] = {
            "hit_id": hit_id,
            "description": description,
            "gene_name": gene_name or SearchUtilities.extract_gene_name(description),
            "organism": organism or UNKNOWN_LABEL,
            "length": length,
            "database": database,
        }

        if score is not None:
            hit["score"] = score
        if e_value is not None:
            hit["e_value"] = e_value
            hit["confidence"] = SearchUtilities.calculate_confidence(e_value)

        hit.update(kwargs)

        return hit

    @staticmethod
    def build_blast_hit_from_alignment(
        alignment: Any, hsp: Any, database: str = "NCBI BLAST"
    ) -> dict[str, Any]:
        gene_name = SearchUtilities.extract_gene_name(alignment.title)
        return SearchUtilities.create_standardized_hit(
            hit_id=getattr(alignment, "hit_id", ""),
            description=alignment.title,
            gene_name=gene_name,
            organism=UNKNOWN_LABEL,
            length=getattr(alignment, "length", 0),
            score=getattr(hsp, "score", 0.0),
            e_value=getattr(hsp, "expect", 1.0),
            database=database,
            identity=getattr(hsp, "identities", None),
            alignment_length=getattr(hsp, "align_length", None),
            query_start=getattr(hsp, "query_start", None),
            query_end=getattr(hsp, "query_end", None),
            subject_start=getattr(hsp, "sbjct_start", None),
            subject_end=getattr(hsp, "sbjct_end", None),
        )

    @staticmethod
    def build_blast_hit_from_tsv(
        fields: list[str], database: str = "NCBI Swiss-Prot"
    ) -> Optional[dict[str, Any]]:
        if len(fields) < SEARCH_TSV_MIN_FIELDS:
            return None
        title = fields[SEARCH_TSV_TITLE_FIELD_INDEX]
        gene_name = SearchUtilities.extract_gene_name(title)
        try:
            return SearchUtilities.create_standardized_hit(
                hit_id=fields[SEARCH_TSV_HIT_ID_INDEX],
                description=title,
                gene_name=gene_name,
                organism=UNKNOWN_LABEL,
                length=int(fields[SEARCH_TSV_LENGTH_INDEX]),
                score=float(fields[SEARCH_TSV_SCORE_INDEX]) if len(fields) > SEARCH_TSV_SCORE_INDEX else 0.0,
                e_value=float(fields[SEARCH_TSV_EVALUE_INDEX]),
                database=database,
                identity=float(fields[SEARCH_TSV_IDENTITY_INDEX]),
                query_start=int(fields[SEARCH_TSV_QUERY_START_INDEX]) if len(fields) > SEARCH_TSV_QUERY_START_INDEX else None,
                query_end=int(fields[SEARCH_TSV_QUERY_END_INDEX]) if len(fields) > SEARCH_TSV_QUERY_END_INDEX else None,
                subject_start=int(fields[SEARCH_TSV_SUBJECT_START_INDEX]) if len(fields) > SEARCH_TSV_SUBJECT_START_INDEX else None,
                subject_end=int(fields[SEARCH_TSV_SUBJECT_END_INDEX]) if len(fields) > SEARCH_TSV_SUBJECT_END_INDEX else None,
            )
        except Exception:
            return None

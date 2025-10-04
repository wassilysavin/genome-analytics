import logging
from typing import Any, Optional

from ...constants.constants import *

from Bio import Entrez

from .base_search import (
    BaseSearchProvider,
    DatabaseType,
    SearchResult,
    SearchType,
    SearchUtilities,
)

logger = logging.getLogger(__name__)


class NCBIProvider(BaseSearchProvider):
    def __init__(self, database_type: DatabaseType, search_type: SearchType):
        super().__init__(database_type, search_type)
        self.config: Optional[Any] = None

    def setup(self) -> bool:
        try:
            from ...settings import settings

            self.config = settings
            self._setup_entrez()
            return True
        except Exception as e:
            self.logger.error(f"Failed to setup NCBI provider: {e}")
            return False

    def _setup_entrez(self) -> None:
        try:
            if not self.config:
                raise RuntimeError("NCBIProvider not configured")
            Entrez.email = self.config.ncbi_email
            if self.config.ncbi_api_key:
                Entrez.api_key = self.config.ncbi_api_key
            self.logger.debug("Entrez configured successfully")
        except Exception as e:
            self.logger.error(f"Failed to configure Entrez: {e}")
            raise

    def search(self, query: str, max_results: int = settings.search_default_max_results) -> SearchResult:
        if not self.validate_query(query):
            return self.create_error_result("Invalid query")

        handlers = {
            SearchType.PROTEIN: self._search_protein,
            SearchType.NUCLEOTIDE: self._search_nucleotide,
            SearchType.GENE_NAME: self._search_gene_name,
        }
        handler = handlers.get(self.search_type)
        if handler is None:
            return self.create_error_result(f"Unsupported search type: {self.search_type}")
        return handler(query, max_results)

    def _search_protein(self, protein_sequence: str, max_results: int) -> SearchResult:
        try:
            search_seq = protein_sequence[:NCBI_DEFAULT_SEQUENCE_LENGTH]

            search_handle = Entrez.esearch(
                db="protein", term=f"{search_seq}[WORD]", retmax=max_results
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()

            if not search_results["IdList"]:
                return SearchResult(
                    found_matches=False,
                    hits=[],
                    search_method="NCBI protein search",
                    database="NCBI Protein",
                )

            if self.config:
                self.rate_limit(self.config.api_delay)
            summaries = self._fetch_entrez_summaries("protein", search_results["IdList"])

            if not summaries:
                return self.create_error_result("Failed to fetch protein summaries")

            hits = []
            for i, summary in enumerate(summaries):
                hit = self._create_ncbi_hit(summary, "protein", i)
                hits.append(hit)

            self.logger.info(f"Found {len(hits)} NCBI protein matches")

            return SearchResult(
                found_matches=True,
                hits=hits,
                total_results=len(hits),
                search_method="NCBI protein search",
                database="NCBI Protein",
            )

        except Exception as e:
            self.logger.error(f"NCBI protein search failed: {e}")
            return self.create_error_result(f"NCBI search error: {str(e)}")

    def _search_nucleotide(self, dna_sequence: str, max_results: int) -> SearchResult:
        try:
            search_seq = dna_sequence[:NCBI_DEFAULT_NUCLEOTIDE_LENGTH]

            search_handle = Entrez.esearch(
                db="nucleotide", term=f"{search_seq}[WORD]", retmax=max_results
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()

            if not search_results["IdList"]:
                return SearchResult(
                    found_matches=False,
                    hits=[],
                    search_method="NCBI nucleotide search",
                    database="NCBI Nucleotide",
                )

            if self.config:
                self.rate_limit(self.config.api_delay)
            summaries = self._fetch_entrez_summaries("nucleotide", search_results["IdList"])

            if not summaries:
                return self.create_error_result("Failed to fetch nucleotide summaries")

            hits = []
            for i, summary in enumerate(summaries):
                hit = self._create_ncbi_hit(summary, "nucleotide", i)
                hits.append(hit)

            self.logger.info(f"Found {len(hits)} NCBI nucleotide matches")

            return SearchResult(
                found_matches=True,
                hits=hits,
                total_results=len(hits),
                search_method="NCBI nucleotide search",
                database="NCBI Nucleotide",
            )

        except Exception as e:
            self.logger.error(f"NCBI nucleotide search failed: {e}")
            return self.create_error_result(f"NCBI nucleotide search error: {str(e)}")

    def _search_gene_name(self, gene_name: str, max_results: int) -> SearchResult:
        try:
            search_handle = Entrez.esearch(
                db="protein", term=f"{gene_name}[Gene Name]", retmax=max_results
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()

            if not search_results["IdList"]:
                return SearchResult(
                    found_matches=False,
                    hits=[],
                    search_method="NCBI gene name search",
                    database="NCBI Protein",
                )

            if self.config:
                self.rate_limit(self.config.api_delay)
            summaries = self._fetch_entrez_summaries("protein", search_results["IdList"])

            if not summaries:
                return self.create_error_result("Failed to fetch gene summaries")

            hits = []
            for i, summary in enumerate(summaries):
                hit = self._create_ncbi_hit(summary, "gene", i)
                hits.append(hit)

            self.logger.info(f"Found {len(hits)} NCBI gene matches for {gene_name}")

            return SearchResult(
                found_matches=True,
                hits=hits,
                total_results=len(hits),
                search_method="NCBI gene name search",
                database="NCBI Protein",
            )

        except Exception as e:
            self.logger.error(f"NCBI gene name search failed: {e}")
            return self.create_error_result(f"NCBI gene name search error: {str(e)}")

    def _fetch_entrez_summaries(self, db: str, id_list: list[str]) -> Optional[list[dict[str, Any]]]:
        try:
            if self.config:
                self.rate_limit(self.config.api_delay)

            summary_handle = Entrez.esummary(db=db, id=",".join(id_list))
            summaries = Entrez.read(summary_handle)
            summary_handle.close()

            return summaries if isinstance(summaries, list) else None

        except Exception as e:
            self.logger.error(f"Entrez summary fetch failed for {db}: {e}")
            return None

    def _create_ncbi_hit(
        self, summary: dict[str, Any], hit_type: str, index: int
    ) -> dict[str, Any]:
        gene_name = None
        if hit_type == "protein":
            gene_name = SearchUtilities.extract_gene_name(summary.get("Title", ""))

            return SearchUtilities.create_standardized_hit(
                hit_id=summary.get("AccessionVersion", UNKNOWN_LABEL),
                description=summary.get("Title", NO_DESCRIPTION),
                gene_name=gene_name,
                organism=summary.get("Organism", UNKNOWN_LABEL),
                length=summary.get("Length", 0),
                database=f"NCBI {hit_type.title()}",
            )


def extract_gene_name(title: str) -> str:
    return SearchUtilities.extract_gene_name(title)

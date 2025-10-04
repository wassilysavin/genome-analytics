import logging
from typing import Any, Optional
from ...constants.constants import *
from ...settings import settings

from Bio import Entrez
from Bio.Blast import NCBIWWW, NCBIXML

from .base_search import (
    BaseSearchProvider,
    DatabaseType,
    SearchResult,
    SearchType,
    SearchUtilities,
)

logger = logging.getLogger(__name__)


class BlastRemoteProvider(BaseSearchProvider):
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
            self.logger.error(f"Failed to setup remote BLAST provider: {e}")
            return False

    def _setup_entrez(self) -> None:
        try:
            if not self.config:
                raise RuntimeError("BlastRemoteProvider not configured")
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

        if self.search_type == SearchType.PROTEIN:
            return self._search_protein(query, max_results)
        elif self.search_type == SearchType.NUCLEOTIDE:
            return self._search_nucleotide(query, max_results)
        else:
            return self.create_error_result(f"Unsupported search type: {self.search_type}")

    def _search_protein(self, protein_sequence: str, max_results: int) -> SearchResult:
        try:
            self.logger.info(
                f"Performing remote BLAST protein search for {len(protein_sequence)} amino acids..."
            )

            result_handle = self._run_blast_query(
                "blastp", "nr", protein_sequence, max_results, BLAST_PROTEIN_WORD_SIZE
            )

            if not result_handle:
                return self.create_error_result("BLAST protein search failed")

            blast_records = NCBIXML.parse(result_handle)
            hits = self._parse_blast_results(blast_records, max_results)

            result_handle.close()

            self.logger.info(f"Remote BLAST protein search completed: {len(hits)} matches found")

            return SearchResult(
                found_matches=len(hits) > 0,
                hits=hits,
                total_results=len(hits),
                search_method="Remote BLAST protein search",
                database="NCBI BLAST",
            )

        except Exception as e:
            self.logger.error(f"Remote BLAST protein search failed: {e}")
            return self.create_error_result(f"BLAST protein search error: {str(e)}")

    def _search_nucleotide(self, dna_sequence: str, max_results: int) -> SearchResult:
        try:
            self.logger.info(
                f"Performing remote BLAST nucleotide search for {len(dna_sequence)} bp..."
            )

            result_handle = self._run_blast_query(
                "blastn", "nt", dna_sequence, max_results, BLAST_NUCLEOTIDE_WORD_SIZE
            )

            if not result_handle:
                return self.create_error_result("BLAST nucleotide search failed")

            blast_records = NCBIXML.parse(result_handle)
            hits = self._parse_blast_results(blast_records, max_results)

            result_handle.close()

            self.logger.info(f"Remote BLAST nucleotide search completed: {len(hits)} matches found")

            return SearchResult(
                found_matches=len(hits) > 0,
                hits=hits,
                total_results=len(hits),
                search_method="Remote BLAST nucleotide search",
                database="NCBI BLAST",
            )

        except Exception as e:
            self.logger.error(f"Remote BLAST nucleotide search failed: {e}")
            return self.create_error_result(f"BLAST nucleotide search error: {str(e)}")

    def _run_blast_query(
        self, program: str, database: str, sequence: str, max_results: int, word_size: int
    ) -> Optional[Any]:
        try:
            self.logger.info(f"Performing {program} search for {len(sequence)} characters...")

            result_handle = NCBIWWW.qblast(
                program,
                database,
                sequence,
                hitlist_size=min(max_results, settings.blast_remote_default_max_results),
                expect=BLAST_EXPECT_THRESHOLD,
                gapcosts=BLAST_GAP_COSTS,
                word_size=word_size,
            )

            self.logger.debug("BLAST query submitted successfully")
            return result_handle

        except Exception as e:
            self.logger.warning(f"BLAST {program} search failed: {e}")
            return None

    def _parse_blast_results(self, blast_records, max_results: int) -> list:
        hits = []
        for blast_record in blast_records:
            for alignment in blast_record.alignments:
                for hsp in alignment.hsps:
                    hit = SearchUtilities.build_blast_hit_from_alignment(
                        alignment, hsp, database="NCBI BLAST"
                    )
                    hits.append(hit)
                    self.logger.info(f"FOUND GENE: {hit.get('gene_name', UNKNOWN_LABEL)}")
                    self.logger.debug(f"   Description: {alignment.title[:LOG_DESC_PREVIEW_LONG]}...")
                    self.logger.debug(f"   E-value: {hsp.expect}")
                    self.logger.debug(f"   Identity: {hsp.identities}/{hsp.align_length}")
                    if len(hits) >= max_results:
                        return hits
        return hits



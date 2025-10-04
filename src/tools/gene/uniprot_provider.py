import logging
from typing import Any, Optional, Union, Mapping, Iterable, Mapping
import requests

from ...constants.constants import *
from ...settings import settings

from .base_search import (
    BaseSearchProvider,
    DatabaseType,
    SearchResult,
    SearchType,
    SearchUtilities,
)

logger = logging.getLogger(__name__)


class UniProtProvider(BaseSearchProvider):
    SEQUENCE_FIELDS = "accession,protein_name,gene_names,organism_name,sequence,length,keyword,go"
    GENE_FIELDS = "accession,protein_name,gene_names,organism_name,function_cc"

    def __init__(self, database_type: DatabaseType, search_type: SearchType):
        super().__init__(database_type, search_type)
        self.config: Optional[Any] = None

    def setup(self) -> bool:
        try:
            from ...settings import settings

            self.config = settings
            return True
        except Exception as e:
            self.logger.error(f"Failed to setup UniProt provider: {e}")
            return False

    def search(self, query: str, max_results: int = settings.search_default_max_results) -> SearchResult:
        if not self.validate_query(query):
            return self.create_error_result("Invalid query")

        if self.search_type == SearchType.PROTEIN:
            return self._search_by_sequence(query, max_results)
        elif self.search_type == SearchType.GENE_NAME:
            return self._search_by_gene_name(query, max_results)
        else:
            return self.create_error_result(f"Unsupported search type: {self.search_type}")

    def _search_by_sequence(self, protein_sequence: str, max_results: int) -> SearchResult:
        try:
            search_seq = protein_sequence[: UNIPROT_DEFAULT_SEQUENCE_LENGTH]
            query = f"sequence:{search_seq}"

            data = self._make_uniprot_request(query, self.SEQUENCE_FIELDS, max_results)
            if not data:
                return self.create_error_result("UniProt API request failed")

            if not data.get("results"):
                return SearchResult(
                    found_matches=False,
                    hits=[],
                    search_method="UniProt sequence search",
                    database="UniProt",
                )

            hits = []
            for i, entry in enumerate(data["results"]):
                hit = self._create_uniprot_hit(entry, "sequence", i, query_sequence=protein_sequence)
                hits.append(hit)

            self.logger.info(f"Found {len(hits)} UniProt matches")

            return SearchResult(
                found_matches=True,
                hits=hits,
                total_results=len(hits),
                search_method="UniProt sequence search",
                database="UniProt",
            )

        except Exception as e:
            self.logger.error(f"UniProt search failed: {e}")
            return self.create_error_result(f"UniProt search error: {str(e)}")

    def _search_by_gene_name(self, gene_name: str, max_results: int) -> SearchResult:
        try:
            query = f"gene:{gene_name}"

            data = self._make_uniprot_request(query, self.GENE_FIELDS, max_results)
            if not data:
                return self.create_error_result("UniProt API request failed")

            if not data.get("results"):
                return SearchResult(
                    found_matches=False,
                    hits=[],
                    search_method="UniProt gene search",
                    database="UniProt",
                )

            hits = []
            for entry in data["results"]:
                hit = self._create_uniprot_hit(entry, "gene", query_gene_name=gene_name)
                hits.append(hit)

            self.logger.info(f"Found {len(hits)} UniProt gene matches for {gene_name}")

            return SearchResult(
                found_matches=True,
                hits=hits,
                total_results=len(hits),
                search_method="UniProt gene search",
                database="UniProt",
            )

        except Exception as e:
            self.logger.error(f"UniProt gene search failed: {e}")
            return self.create_error_result(f"UniProt gene search error: {str(e)}")

    def _make_uniprot_request(self, query: str, fields: str, limit: int) -> Optional[dict[str, Any]]:
        try:
            params: Mapping[str, Union[str, int]] = {
                "query": query,
                "format": "json",
                "limit": limit,
                "fields": fields,
            }

            timeout = DEFAULT_REQUEST_TIMEOUT
            if self.config and getattr(self.config, "request_timeout", None):
                try:
                    timeout = int(self.config.request_timeout)
                except Exception:
                    timeout = DEFAULT_REQUEST_TIMEOUT
            response = requests.get(UNIPROT_BASE_URL, params=params, timeout=timeout)
            response.raise_for_status()

            data = response.json()
            if isinstance(data, dict):
                return data
            return None

        except requests.RequestException as e:
            self.logger.error(f"UniProt API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"UniProt request error: {e}")
            return None

    def _create_uniprot_hit(
        self,
        entry: dict[str, Any],
        hit_type: str = "sequence",
        index: int = 0,
        query_sequence: Optional[str] = None,
        query_gene_name: Optional[str] = None,
    ) -> dict[str, Any]:
        protein_name = self._extract_protein_name(entry)
        gene_names = self._extract_gene_names(entry)
        organism = entry.get("organism", {}).get("scientificName", UNKNOWN_LABEL)

        if hit_type == "sequence":
            score_val = self._calculate_sequence_similarity_score(query_sequence, entry)
            return SearchUtilities.create_standardized_hit(
                hit_id=entry.get("primaryAccession", UNKNOWN_LABEL),
                description=protein_name,
                gene_name=gene_names[0] if gene_names else UNKNOWN_LABEL,
                organism=organism,
                length=entry.get("sequence", {}).get("length", 0),
                score=score_val,
                database="UniProt",
                keywords=self._extract_keywords(entry),
                go_terms=self._extract_go_terms(entry),
            )
        else: 
            score_val = self._calculate_gene_name_similarity_score(query_gene_name, gene_names, protein_name)
            return SearchUtilities.create_standardized_hit(
                hit_id=entry.get("primaryAccession", UNKNOWN_LABEL),
                description=protein_name,
                gene_name=gene_names[0] if gene_names else UNKNOWN_LABEL,
                organism=organism,
                length=0, 
                score=score_val,
                database="UniProt",
                function=self._extract_function(entry),
            )

    def _extract_protein_name(self, entry: dict) -> str:
        try:
            protein_description = entry.get("proteinDescription", {})
            recommended_name = protein_description.get("recommendedName", {})

            if recommended_name and "fullName" in recommended_name:
                full = recommended_name["fullName"].get("value", UNKNOWN_PROTEIN)
                return str(full)

            alternative_names = protein_description.get("alternativeNames", [])
            if alternative_names:
                alt = alternative_names[0].get("fullName", {}).get("value", UNKNOWN_PROTEIN)
                return str(alt)

            return UNKNOWN_PROTEIN

        except Exception:
            return UNKNOWN_PROTEIN

    def _extract_gene_names(self, entry: dict) -> list[str]:
        try:
            gene_names = []

            genes = entry.get("genes", [])
            for gene in genes:
                if "geneName" in gene:
                    gene_names.append(gene["geneName"]["value"])

                synonyms = gene.get("synonyms", [])
                for synonym in synonyms:
                    gene_names.append(synonym["value"])

            return list(set(gene_names))  

        except Exception:
            return [UNKNOWN_LABEL]

    def _extract_keywords(self, entry: dict) -> list[str]:
        try:
            keywords = entry.get("keywords", [])
            return [kw.get("name", "") for kw in keywords if kw.get("name")]

        except Exception:
            return []

    def _extract_go_terms(self, entry: dict) -> list[str]:
        try:
            go_terms = []
            dbreferences = entry.get("uniProtKBCrossReferences", [])

            for dbref in dbreferences:
                if dbref.get("database") == "GO":
                    properties = dbref.get("properties", [])
                    for prop in properties:
                        if prop.get("key") == "GoTerm":
                            go_terms.append(prop.get("value", ""))

            return go_terms

        except Exception:
            return []

    def _extract_function(self, entry: dict) -> str:
        try:
            comments = entry.get("comments", [])

            for comment in comments:
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        return str(texts[0].get("value", UNKNOWN_FUNCTION))

            return UNKNOWN_FUNCTION

        except Exception:
            return UNKNOWN_FUNCTION

    def _calculate_sequence_similarity_score(self, query_sequence: Optional[str], entry: dict[str, Any]) -> Optional[float]:
        try:
            if not query_sequence:
                return None
            qs = query_sequence.replace("*", "").upper()
            if not qs:
                return None
            seq_obj = entry.get("sequence", {})
            target_seq = str(seq_obj.get("value", "")).replace("*", "").upper()
            if not target_seq:
                return None

            qlen = len(qs)
            tlen = len(target_seq)
            if qlen == 0 or tlen == 0:
                return None

            # Sliding window best identity over target
            window = min(qlen, tlen)
            best_matches = 0
            # Compare against all windows of length 'window'
            for i in range(0, tlen - window + 1):
                matches = 0
                segment = target_seq[i : i + window]
                for a, b in zip(qs[:window], segment):
                    if a == b:
                        matches += 1
                if matches > best_matches:
                    best_matches = matches

            similarity = (best_matches / window) * 100.0
            return round(similarity, 2)
        except Exception:
            return None

    def _calculate_gene_name_similarity_score(
        self, query_gene_name: Optional[str], gene_names: list[str], protein_name: str
    ) -> Optional[float]:
        try:
            if not query_gene_name:
                return None
            import difflib

            candidates: list[str] = list(gene_names) if gene_names else []
            if protein_name:
                candidates.append(protein_name)

            q = query_gene_name.upper()
            best_ratio = 0.0
            for cand in candidates:
                r = difflib.SequenceMatcher(None, q, str(cand).upper()).ratio()
                if r > best_ratio:
                    best_ratio = r
            return round(best_ratio * 100.0, 2) if best_ratio > 0 else None
        except Exception:
            return None


def rate_limit():
    import time

    from ...settings import settings

    time.sleep(settings.api_delay)

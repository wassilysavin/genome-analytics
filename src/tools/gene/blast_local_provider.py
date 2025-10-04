import gzip
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Optional

from .base_search import (
    BaseSearchProvider,
    DatabaseType,
    SearchResult,
    SearchType,
    SearchUtilities,
)
from ...constants.constants import *
from ...settings import settings

logger = logging.getLogger(__name__)


class BlastLocalProvider(BaseSearchProvider):
    BLAST_CONFIG = {
        "database_dir": "blast_databases",
        "database_name": "swissprot",
        "swissprot_url": SWISSPROT_FASTA_URL,
        "max_results": 5,
        "evalue_threshold": 10,
        "num_threads": 4,
        "word_size": 3,
        "threshold": 11,
    }

    def __init__(self, database_type: DatabaseType, search_type: SearchType):
        super().__init__(database_type, search_type)
        self.config: Optional[Any] = None

    def setup(self) -> bool:
        try:
            from ...settings import settings

            self.config = settings
            return True
        except Exception as e:
            self.logger.error(f"Failed to setup local BLAST provider: {e}")
            return False

    def search(self, query: str, max_results: int = settings.search_default_max_results) -> SearchResult:
        if not self.validate_query(query):
            return self.create_error_result("Invalid query")

        if self.search_type != SearchType.PROTEIN:
            return self.create_error_result(
                f"Local BLAST only supports protein search, got: {self.search_type}"
            )

        try:
            db_path = self._ensure_database_available()
            if not db_path:
                return self.create_error_result("Failed to download NCBI database")

            blast_result = self._run_blast_search(query, db_path, max_results)

            return blast_result

        except subprocess.CalledProcessError as e:
            self.logger.error(f"BLAST command failed: {e}")
            return self.create_error_result(f"BLAST command failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Local BLAST failed: {e}")
            return self.create_error_result(f"Local BLAST failed: {str(e)}")

    def _ensure_database_available(self) -> Optional[str]:
        db_dir_val = str(self.BLAST_CONFIG["database_dir"])  
        db_name_val = str(self.BLAST_CONFIG["database_name"]) 
        db_name = Path(db_dir_val) / db_name_val
        db_files = [
            db_name.with_suffix(".phr"),
            db_name.with_suffix(".pin"),
            db_name.with_suffix(".psq"),
        ]

        if not all(f.exists() for f in db_files):
            self.logger.info("BLAST database not found, downloading...")
            return self._download_ncbi_database()
        else:
            self.logger.info("Using existing BLAST database")
            return str(db_name)

    def _download_ncbi_database(self) -> Optional[str]:
        self.logger.info("Downloading NCBI protein database subset...")

        try:
            db_dir = Path(str(self.BLAST_CONFIG["database_dir"]))
            db_dir.mkdir(exist_ok=True)

            local_file = db_dir / f"{self.BLAST_CONFIG['database_name']}.gz"
            fasta_file = db_dir / f"{self.BLAST_CONFIG['database_name']}.fasta"

            self._download_and_extract_database(
                str(self.BLAST_CONFIG["swissprot_url"]), local_file, fasta_file
            )

            db_path = self._create_blast_database(
                fasta_file, db_dir / str(self.BLAST_CONFIG["database_name"])
            )

            return db_path

        except Exception as e:
            self.logger.error(f"Failed to download NCBI database: {e}")
            return None

    def _download_and_extract_database(self, url: str, local_file: Path, fasta_file: Path) -> None:
        self.logger.info("Downloading Swiss-Prot database...")
        urllib.request.urlretrieve(url, local_file)

        self.logger.info("Extracting database...")
        with gzip.open(local_file, "rb") as f_in, open(fasta_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        local_file.unlink()

        file_size_mb = fasta_file.stat().st_size / 1024 / 1024
        self.logger.info(f"Downloaded NCBI Swiss-Prot database: {file_size_mb:.1f} MB")

    def _create_blast_database(self, fasta_file: Path, db_name: Path) -> str:
        self.logger.info("Creating BLAST database format...")
        cmd = ["makeblastdb", "-in", str(fasta_file), "-dbtype", "prot", "-out", str(db_name)]
        subprocess.run(cmd, check=True, capture_output=True)

        self.logger.info(f"Created BLAST database: {db_name}")
        return str(db_name)

    def _run_blast_search(
        self, protein_sequence: str, db_path: str, max_results: int
    ) -> SearchResult:

        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as temp_file:
            temp_file.write(f">query\n{protein_sequence}\n")
            temp_fasta = temp_file.name

        try:
            self.logger.info(
                f"Running local BLAST search against NCBI database "
                f"for {len(protein_sequence)} amino acids..."
            )

            cmd = self._build_blast_command(temp_fasta, db_path, max_results)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            hits = self._parse_blast_output(result.stdout)

            self.logger.info(f"Local BLAST completed: {len(hits)} matches found")

            return SearchResult(
                found_matches=len(hits) > 0,
                hits=hits,
                total_results=len(hits),
                search_method="Local BLAST against NCBI Swiss-Prot",
                database="NCBI Swiss-Prot",
            )

        finally:
            os.unlink(temp_fasta)

    def _build_blast_command(self, temp_fasta: str, db_path: str, max_results: int) -> list[str]:
        return [
            "blastp",
            "-query",
            temp_fasta,
            "-db",
            db_path,
            "-outfmt",
            "6 qseqid sseqid pident length mismatch gapopen "
            "qstart qend sstart send evalue bitscore stitle",
            "-max_target_seqs",
            str(max_results),
            "-evalue",
            str(self.BLAST_CONFIG["evalue_threshold"]),
            "-num_threads",
            str(self.BLAST_CONFIG["num_threads"]),
            "-word_size",
            str(self.BLAST_CONFIG["word_size"]),
            "-threshold",
            str(self.BLAST_CONFIG["threshold"]),
        ]

    def _parse_blast_output(self, output: str) -> list[dict[str, Any]]:
        hits = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            fields = line.split("\t")
            hit = SearchUtilities.build_blast_hit_from_tsv(fields, database="NCBI Swiss-Prot")
            if hit:
                hits.append(hit)
        return hits

import logging
import math
from typing import Any
from ...settings import settings
from ...constants.constants import *
from langchain.tools import tool
from ...models.chunking_models import Chunk, ChunkResult, ProcessingStats

logger = logging.getLogger(__name__)


class SequenceChunker:
    def __init__(self, chunk_size: int = None, overlap_size: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap_size = overlap_size or settings.chunk_overlap

    def chunk_sequence(self, sequence: str) -> list[Chunk]:
        sequence_length = len(sequence)

        if sequence_length <= self.chunk_size:
            return [
                Chunk(
                    chunk_id=1,
                    start=0,
                    end=sequence_length,
                    sequence=sequence,
                    is_complete=True,
                    total_chunks=1,
                )
            ]

        step_size = max(1, self.chunk_size - self.overlap_size)
        total_chunks = math.ceil((sequence_length - self.overlap_size) / step_size)

        chunks = []
        for i in range(total_chunks):
            start = i * step_size
            end = min(start + self.chunk_size, sequence_length)

            if i == total_chunks - 1:
                end = sequence_length

            chunk_sequence = sequence[start:end]

            chunks.append(
                Chunk(
                    chunk_id=i + 1,
                    start=start,
                    end=end,
                    sequence=chunk_sequence,
                    is_complete=False,
                    total_chunks=total_chunks,
                    overlap_start=start > 0,
                    overlap_end=end < sequence_length,
                )
            )

        logger.info(f"Created {len(chunks)} chunks from {sequence_length:,} bp sequence")
        return chunks

    def combine_chunk_results(
        self, chunk_results: list[dict[str, Any]], original_length: int
    ) -> dict[str, Any]:
        stats = ProcessingStats()
        all_orfs: list[dict[str, Any]] = []
        database_hits = self._initialize_database_hits()
        chunk_summaries = []

        for i, chunk_result in enumerate(chunk_results):
            chunk_id = i + 1
            chunk_summary = self._process_chunk_result(chunk_result, chunk_id)
            chunk_summaries.append(chunk_summary)

            if chunk_summary.success:
                stats.successful_chunks += 1
                self._extract_orfs_from_chunk(chunk_result, chunk_id, all_orfs)
                self._extract_database_hits_from_chunk(chunk_result, chunk_id, database_hits)
            else:
                stats.failed_chunks += 1

        unique_orfs = self._deduplicate_orfs(all_orfs)

        return {
            "success": True,
            "original_sequence_length": original_length,
            "total_chunks_processed": len(chunk_results),
            "chunking_method": f"Smart chunking ({self.chunk_size} bp chunks, {self.overlap_size} bp overlap)",
            "combined_analysis": {
                "total_orfs_found": len(unique_orfs),
                "unique_orfs": unique_orfs,
                "database_results": database_hits,
                "total_database_hits": sum(len(hits) for hits in database_hits.values()),
                "processing_stats": {
                    "successful_chunks": stats.successful_chunks,
                    "failed_chunks": stats.failed_chunks,
                    "success_rate": stats.success_rate,
                },
            },
            "chunk_summaries": [summary.__dict__ for summary in chunk_summaries],
        }

    def _initialize_database_hits(self) -> dict[str, list]:
        return {"ncbi_protein_results": [], "ncbi_nucleotide_results": [], "uniprot_results": []}

    def _process_chunk_result(self, chunk_result: dict[str, Any], chunk_id: int) -> ChunkResult:
        if not chunk_result.get("success", False):
            return ChunkResult(
                chunk_id=chunk_id, success=False, error=chunk_result.get("error", "Unknown error")
            )

        orfs_found = self._count_orfs_in_chunk(chunk_result)
        blast_hits = self._count_blast_hits_in_chunk(chunk_result)

        return ChunkResult(
            chunk_id=chunk_id,
            success=True,
            orfs_found=orfs_found,
            blast_hits=blast_hits,
            chunk_length=chunk_result.get("sequence_length", 0),
        )

    def _count_orfs_in_chunk(self, chunk_result: dict[str, Any]) -> int:
        if (
            "sequence_analysis" in chunk_result
            and "analyzed_orfs" in chunk_result["sequence_analysis"]
        ):
            return len(chunk_result["sequence_analysis"]["analyzed_orfs"])
        return 0

    def _count_blast_hits_in_chunk(self, chunk_result: dict[str, Any]) -> int:
        if "database_results" not in chunk_result:
            return 0

        db_results = chunk_result["database_results"]
        return (
            len(db_results.get("ncbi_protein_results", []))
            + len(db_results.get("ncbi_nucleotide_results", []))
            + len(db_results.get("uniprot_results", []))
        )

    def _extract_orfs_from_chunk(
        self, chunk_result: dict[str, Any], chunk_id: int, all_orfs: list[dict[str, Any]]
    ):
        if "sequence_analysis" not in chunk_result:
            return

        seq_analysis = chunk_result["sequence_analysis"]
        if "analyzed_orfs" not in seq_analysis:
            return

        chunk_start = chunk_result.get("chunk_start", 0)

        for orf in seq_analysis["analyzed_orfs"]:
            if not isinstance(orf, dict):
                logger.warning(f"Skipping invalid ORF format in chunk {chunk_id}: {type(orf)}")
                continue

            global_orf = self._adjust_orf_positions(orf, chunk_start, chunk_id)
            all_orfs.append(global_orf)

    def _adjust_orf_positions(
        self, orf: dict[str, Any], chunk_start: int, chunk_id: int
    ) -> dict[str, Any]:
        global_orf = orf.copy()

        if "genomic_position" in orf and isinstance(orf["genomic_position"], dict):
            pos = orf["genomic_position"]
            global_orf["genomic_position"] = {
                "start": pos.get("start", 0) + chunk_start,
                "end": pos.get("end", 0) + chunk_start,
                "strand": pos.get("strand", "+"),
                "frame": pos.get("frame", 1),
                "chunk_id": chunk_id,
            }
        else:
            global_orf["genomic_position"] = {
                "start": chunk_start,
                "end": chunk_start + orf.get("length", 0),
                "strand": orf.get("strand", "+"),
                "frame": orf.get("frame", 1),
                "chunk_id": chunk_id,
            }

        return global_orf

    def _extract_database_hits_from_chunk(
        self, chunk_result: dict[str, Any], chunk_id: int, database_hits: dict[str, list]
    ):
        if "database_results" not in chunk_result:
            logger.warning(f"Chunk {chunk_id}: No database_results found")
            return

        db_results = chunk_result["database_results"]

        for db_type in ["ncbi_protein_results", "ncbi_nucleotide_results", "uniprot_results"]:
            if db_type in db_results and db_results[db_type]:
                for hit in db_results[db_type]:
                    if isinstance(hit, dict):
                        hit_copy = hit.copy()
                        hit_copy["source_chunk"] = chunk_id
                        database_hits[db_type].append(hit_copy)
            else:
                logger.debug(f"Chunk {chunk_id}: No {db_type} found or empty")

    def _deduplicate_orfs(self, orfs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not orfs:
            return []

        valid_orfs = self._normalize_orfs(orfs)
        if not valid_orfs:
            return []

        sorted_orfs = sorted(valid_orfs, key=self._get_orf_start_position)

        unique_orfs = self._remove_duplicate_orfs(sorted_orfs)

        logger.info(f"Deduplicated ORFs: {len(orfs)} -> {len(unique_orfs)}")
        return unique_orfs

    def _normalize_orfs(self, orfs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid_orfs = []

        for orf in orfs:
            if not isinstance(orf, dict):
                logger.warning(f"Skipping invalid ORF format: {type(orf)}")
                continue

            if "genomic_position" not in orf:
                logger.warning(f"Skipping ORF without position data: {orf}")
                continue

            if not isinstance(orf["genomic_position"], dict):
                logger.warning(
                    f"Skipping ORF with invalid genomic_position: {type(orf['genomic_position'])}"
                )
                continue

            valid_orfs.append(orf)

        return valid_orfs

    def _get_orf_start_position(self, orf: dict[str, Any]) -> int:
        pos = orf.get("genomic_position", {})
        start = pos.get("start", 0)
        return int(start)

    def _remove_duplicate_orfs(self, sorted_orfs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique_orfs: list[dict[str, Any]] = []
        tolerance = ORF_DEDUP_TOLERANCE_BP  # bp

        for orf in sorted_orfs:
            is_duplicate = False
            pos = orf["genomic_position"]

            for existing_orf in unique_orfs:
                existing_pos = existing_orf["genomic_position"]

                # Check for overlap
                if (
                    abs(pos.get("start", 0) - existing_pos.get("start", 0)) <= tolerance
                    and abs(pos.get("end", 0) - existing_pos.get("end", 0)) <= tolerance
                    and pos.get("strand") == existing_pos.get("strand")
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_orfs.append(orf)

        return unique_orfs


sequence_chunker = SequenceChunker()


@tool
def chunk_large_sequence(
    sequence: str, chunk_size: int = None, overlap_size: int = None
) -> dict[str, Any]:
    """
    Split a large DNA sequence into manageable chunks for analysis.

    Args:
        sequence: DNA sequence to chunk
        chunk_size: Size of each chunk in base pairs
        overlap_size: Overlap between chunks in base pairs

    Returns:
        Dictionary with chunking information and chunk list
    """
    try:
        chunk_size = chunk_size or settings.chunk_size
        overlap_size = overlap_size or settings.chunk_overlap
        logger.info(f"Chunking sequence ({len(sequence):,} bp) into {chunk_size} bp chunks...")

        chunker = SequenceChunker(chunk_size, overlap_size)
        chunks = chunker.chunk_sequence(sequence)

        logger.info(f"Created {len(chunks)} chunks")

        return {
            "success": True,
            "original_length": len(sequence),
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "chunks": [chunk.__dict__ for chunk in chunks],
            "chunking_strategy": "Overlapping chunks for gene boundary detection",
        }

    except Exception as e:
        logger.error(f"Sequence chunking failed: {e}")
        return {"success": False, "error": str(e)}


@tool
def process_chunked_sequence(sequence: str, chunk_size: int = None) -> dict[str, Any]:
    """
    Process a large sequence using smart chunking and combine results.

    Args:
        sequence: DNA sequence to analyze
        chunk_size: Size of each chunk in base pairs

    Returns:
        Dictionary with combined analysis results
    """
    try:
        logger.info(f"Processing large sequence ({len(sequence):,} bp) with smart chunking...")

        chunk_size = chunk_size or settings.chunk_size
        overlap_size = settings.chunk_overlap

        # Check if chunking is needed
        if len(sequence) <= chunk_size:
            logger.info("Sequence small enough, processing directly...")
            from ..gene.gene_identification import _search_databases_for_genes_internal

            return _search_databases_for_genes_internal(sequence, CHUNK_SEARCH_MAX_RESULTS)

        # Create chunks and process them
        chunker = SequenceChunker(chunk_size, overlap_size=overlap_size)
        chunks = chunker.chunk_sequence(sequence)
        chunk_results = _process_chunks(chunks)

        # Combine results
        logger.info("Combining chunk results...")
        combined_result = chunker.combine_chunk_results(chunk_results, len(sequence))

        total_orfs = combined_result.get("combined_analysis", {}).get("total_orfs_found", 0)
        logger.info(f"Chunked analysis completed: {total_orfs} unique ORFs found")

        return combined_result

    except Exception as e:
        logger.error(f"Chunked sequence processing failed: {e}")
        return _create_error_result(sequence, chunk_size, str(e))


def _process_chunks(chunks: list[Chunk]) -> list[dict[str, Any]]:
    chunk_results = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        logger.info(
            f"Processing chunk {chunk.chunk_id}/{total_chunks} ({chunk.length:,} bp)... [{i+1}/{total_chunks}]"
        )

        try:
            from ..gene.gene_identification import _search_databases_for_genes_internal

            chunk_result = _search_databases_for_genes_internal(chunk.sequence, 100)
            chunk_result["chunk_start"] = chunk.start
            chunk_result["chunk_id"] = chunk.chunk_id
            chunk_results.append(chunk_result)

        except Exception as e:
            logger.error(f"Chunk {chunk.chunk_id} processing failed: {e}")
            chunk_results.append({"success": False, "error": str(e), "chunk_id": chunk.chunk_id})

    return chunk_results


def _create_error_result(sequence: str, chunk_size: int, error: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": error,
        "original_sequence_length": len(sequence),
        "chunking_method": f"Smart chunking ({chunk_size} bp chunks, {settings.chunk_overlap} bp overlap)",
        "combined_analysis": {
            "total_orfs_found": 0,
            "unique_orfs": [],
            "database_results": {
                "ncbi_protein_results": [],
                "ncbi_nucleotide_results": [],
                "uniprot_results": [],
            },
            "total_database_hits": 0,
            "processing_stats": {"successful_chunks": 0, "failed_chunks": 1, "success_rate": "0%"},
        },
    }

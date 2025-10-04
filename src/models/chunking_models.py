from typing import Optional, Union
from dataclasses import dataclass

from ..constants.constants import *


@dataclass
class Chunk:
    chunk_id: int
    start: int
    end: int
    sequence: str
    is_complete: bool = False
    total_chunks: int = 1
    overlap_start: bool = False
    overlap_end: bool = False

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class ChunkResult:
    chunk_id: int
    success: bool
    orfs_found: int = 0
    blast_hits: int = 0
    error: Optional[str] = None
    chunk_length: int = 0


@dataclass
class ProcessingStats:
    successful_chunks: int = 0
    failed_chunks: int = 0

    @property
    def success_rate(self) -> str:
        total = self.successful_chunks + self.failed_chunks
        if total == 0:
            return "0%"
        return f"{(self.successful_chunks / total * PERCENTAGE_MULTIPLIER):.1f}%"

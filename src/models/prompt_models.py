from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..constants.constants import *


class PromptCategory(Enum):
    GROK = "grok"
    GENE_ANALYSIS = "gene_analysis"
    RAG = "rag"
    GENE_CONTEXT = "gene_context"
    ERROR_RESPONSES = "error_responses"
    PINECONE_RAG = "pinecone_rag"
    GENE_IDENTIFICATION = "gene_identification"
    GENE_AGENT = "gene_agent"
    INSIGHTS = "insights"


@dataclass
class PromptTemplate:
    name: str
    category: PromptCategory
    template: str
    description: str
    variables: list[str]
    max_length: Optional[int] = None
    min_length: Optional[int] = None


@dataclass
class PromptConfig:
    max_response_length: int = DEFAULT_PROMPT_MAX_RESPONSE_LENGTH
    min_response_length: int = DEFAULT_PROMPT_MIN_RESPONSE_LENGTH
    include_gene_context: bool = True
    include_personalization: bool = True
    default_concise_mode: bool = True


@dataclass
class GeneContext:
    gene_count: int
    gene_list: list[str]
    detailed_gene_info: Optional[str] = None
    has_genes: bool = True


@dataclass
class PromptVariables:
    user_question: str
    gene_context: Optional[GeneContext] = None
    sequence: Optional[str] = None
    analysis_data: Optional[dict[str, Any]] = None
    database_search_results: Optional[dict[str, Any]] = None
    context: Optional[str] = None
    gene_name: Optional[str] = None
    base_response: Optional[str] = None
    original_prompt: Optional[str] = None
    user_message: Optional[str] = None
    genome_summary: Optional[dict[str, Any]] = None
    annotation_data: Optional[dict[str, Any]] = None

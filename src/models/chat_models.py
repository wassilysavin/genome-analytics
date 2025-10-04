from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..constants.constants import *


class ResponseMode(Enum):
    RAG = "rag"
    GENERAL = "general"
    ERROR = "error"


class ResponseType(Enum):
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ChatRequest:
    prompt: str
    gene_names: list[str]
    conversation_history: Optional[list[dict[str, str]]] = None
    strict_mode: bool = False
    min_relevance: float = CHAT_DEFAULT_MIN_RELEVANCE


@dataclass
class ChatResponse:
    content: str
    response_type: ResponseType
    mode: ResponseMode
    gene_context_used: bool = False
    error: Optional[str] = None


@dataclass
class ChatGeneContext:
    gene_names: list[str]
    gene_count: int
    has_genes: bool
    context_text: str


@dataclass
class ConversationMessage:
    role: str 
    content: str
    timestamp: Optional[str] = None

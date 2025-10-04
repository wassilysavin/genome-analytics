from typing import Optional, Union
import json
import logging
from ..constants.constants import *
from ..settings import settings

from .llm_factory import create_llm
from ..models.chat_models import (
    ChatRequest,
)
from ..prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


INSUFFICIENT_CONTEXT_PHRASES = [
    "provided context",
    "does not contain",
    "not have the necessary information",
    "no relevant genome data",
    "context from your genome analysis",
]

GENE_KEYWORDS = [
    "gene",
    "protein",
    "sequence",
    "function",
    "role",
    "expression",
    "mutation",
    "variant",
    "allele",
    "genotype",
    "phenotype",
    "disease",
    "cancer",
    "trait",
    "inheritance",
    "pathway",
]


class ChatManager:
    def __init__(self, rag_system: Optional[object] = None) -> None:
        self.rag_system = rag_system
        self.prompt_manager = PromptManager()
        self.llm = None

    def get_chat_response(
        self,
        prompt: str,
        gene_names: list[str],
        conversation_history: Optional[list[dict[str, str]]] = None,
        strict_mode: bool = None,
        min_relevance: float = CHAT_DEFAULT_MIN_RELEVANCE,
    ) -> str:
        try:
            if strict_mode is None:
                strict_mode = settings.default_strict_mode
                
            request = ChatRequest(
                prompt=prompt,
                gene_names=gene_names,
                conversation_history=conversation_history,
                strict_mode=strict_mode,
                min_relevance=min_relevance,
            )
            if self.rag_system is not None and self._is_gene_specific_question(request):
                try:
                    response = self._get_rag_response(request)
                    if isinstance(response, dict):
                        has_context_val = bool(response.get("has_context"))
                        text_val = str(response.get("text", ""))
                        if has_context_val and text_val:
                            return text_val
                    elif isinstance(response, str) and response.strip():
                        return response
                except Exception:
                    pass
            return self._get_general_response(request)
        except Exception as e:
            logger.error(f"Chat response failed: {e}")
            return f"Error: Failed to generate response. {str(e)}"

    def _is_gene_specific_question(self, request: ChatRequest) -> bool:
        prompt_lower = request.prompt.lower()
        for gene_name in request.gene_names:
            if gene_name.lower() in prompt_lower:
                return True
        return bool(any(keyword in prompt_lower for keyword in GENE_KEYWORDS) and request.gene_names)

    def _get_rag_response(self, request: ChatRequest) -> Optional[str]:
        try:
            result = self.rag_system.chat(request.prompt)
            text = None
            if hasattr(result, "text"):
                text = result.text
            elif isinstance(result, dict) and "text" in result:
                text = result.get("text")
            elif isinstance(result, str):
                text = result
            if text:
                if self._is_insufficient_rag_text(text):
                    return None
            return text
        except Exception as e:
            logger.warning(f"RAG response failed: {e}")
            return None

    def _is_insufficient_rag_text(self, text: str) -> bool:
        lowered = text.lower()
        return any(phrase in lowered for phrase in INSUFFICIENT_CONTEXT_PHRASES)

    def _get_general_response(self, request: ChatRequest) -> str:
        try:
            if not self.llm:
                self.llm = create_llm()
            gene_context = self._build_gene_context(request.gene_names)
            grok_prompt = self.prompt_manager.format_grok_general_prompt(
                user_question=request.prompt,
                gene_names=request.gene_names,
                gene_context=gene_context,
                conversation_history=request.conversation_history,
            )
            llm_instance = self.llm
            response = llm_instance.invoke(grok_prompt)
            return self._extract_clean_text(response)
        except Exception as e:
            logger.error(f"General response failed: {e}")
            return f"Error: Failed to generate response. {str(e)}"

    def _build_gene_context(self, gene_names: list[str]) -> str:
        if not gene_names:
            return self.prompt_manager.format_gene_context([], "")

        return self.prompt_manager.format_gene_context(gene_names, "")

    def _extract_clean_text(self, response) -> str:
        try:
            if isinstance(response, str):
                return response

            if hasattr(response, "content"):
                return str(response.content)

            if isinstance(response, dict):
                content = response.get("content", "")
                if isinstance(content, str) and content.startswith("{"):
                    try:
                        import json

                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "content" in parsed:
                            return str(parsed["content"])
                        return str(content)
                    except json.JSONDecodeError:
                        return str(content)
                return str(content)

            if isinstance(response, list):
                return " ".join(str(item) for item in response)

            return str(response)

        except Exception as e:
            logger.warning(f"Error extracting clean text: {e}")
            return str(response)

    def get_conversation_summary(self, conversation_history: list[dict[str, str]]) -> str:
        try:
            if not conversation_history:
                return "No conversation history available."
            user_messages = sum(1 for msg in conversation_history if msg.get("role") == "user")
            assistant_messages = sum(
                1 for msg in conversation_history if msg.get("role") == "assistant"
            )

            return f"Conversation summary: {user_messages} user messages, {assistant_messages} assistant responses."

        except Exception as e:
            logger.warning(f"Failed to generate conversation summary: {e}")
            return "Unable to generate conversation summary."

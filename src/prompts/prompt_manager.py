import logging
from typing import Optional

from ..constants.constants import *
from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class PromptManager:
    def __init__(self):
        self._prompt_cache = {}

    def format_grok_general_prompt(
        self,
        user_question: str,
        gene_names: list[str],
        gene_context: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> str:
        history_text = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-PROMPT_RECENT_HISTORY_LIMIT:]
            history_lines = []
            for msg in recent_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    history_lines.append(f"User: {content}")
                elif role == "assistant":
                    history_lines.append(f"Assistant: {content}")

            if history_lines:
                history_text = "\n".join(history_lines)

        return PromptTemplates.get_grok_general_prompt().format(
            gene_context=gene_context,
            conversation_history=history_text,
            user_question=user_question,
        )

    def format_grok_enhancement_prompt(
        self, base_response: str, original_prompt: str, gene_names: list[str]
    ) -> str:
        gene_list = ", ".join(gene_names[:PROMPT_GENE_LIST_MEDIUM])
        return PromptTemplates.get_grok_enhancement_prompt().format(
            base_response=base_response, original_prompt=original_prompt, gene_names=gene_list
        )

    def format_gene_summary_prompt(self, gene_name: str) -> str:
        return PromptTemplates.get_gene_summary_prompt().format(gene_name=gene_name)

    def format_gene_identification_prompt(self, sequence: str) -> str:
        return PromptTemplates.get_gene_identification_prompt().format(sequence=sequence)

    def format_rag_general_prompt(self, user_question: str) -> str:
        return PromptTemplates.get_rag_general_prompt().format(user_question=user_question)

    def format_rag_gene_specific_prompt(self, gene_name: str, user_question: str) -> str:
        return PromptTemplates.get_rag_gene_specific_prompt().format(
            gene_name=gene_name, user_question=user_question
        )

    def format_gene_context(self, gene_names: list[str], detailed_gene_info: str = "") -> str:
        if not gene_names:
            return PromptTemplates.get_no_genes_context()

        gene_count = len(gene_names)
        gene_list = ", ".join(gene_names[:PROMPT_GENE_LIST_LONG])

        return PromptTemplates.get_gene_context_template().format(
            gene_count=gene_count, gene_list=gene_list, detailed_gene_info=detailed_gene_info
        )

    def format_error_response(self, gene_names: list[str]) -> str:
        if gene_names:
            gene_count = len(gene_names)
            gene_list = ", ".join(gene_names[:PROMPT_GENE_LIST_SHORT])
            return PromptTemplates.get_error_with_genes().format(
                gene_count=gene_count, gene_list=gene_list
            )
        else:
            return PromptTemplates.get_error_no_genes()

    def format_personalized_note(self, gene_names: list[str]) -> str:
        gene_list = ", ".join(gene_names[:PROMPT_GENE_LIST_SHORT])
        return PromptTemplates.get_personalized_note_template().format(gene_list=gene_list)

    def format_rag_personalized_note(self, gene_names: list[str]) -> str:
        gene_count = len(gene_names)
        gene_list = ", ".join(gene_names[:PROMPT_GENE_LIST_SHORT])
        return PromptTemplates.get_rag_personalized_note().format(
            gene_count=gene_count, gene_list=gene_list
        )

    def format_pinecone_results_prompt(self, context: str, user_message: str) -> str:
        return PromptTemplates.get_pinecone_results_prompt().format(
            context=context, user_message=user_message
        )

    def format_gene_identification_summary_prompt(self, database_search_results: str) -> str:
        return PromptTemplates.get_gene_identification_summary_prompt().format(
            database_search_results=database_search_results
        )

    def format_gene_agent_prompt(self) -> str:
        return PromptTemplates.get_gene_agent_prompt()

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        try:
            if prompt_name == "grok_general":
                return self.format_grok_general_prompt(
                    kwargs.get("user_question", ""),
                    kwargs.get("gene_names", []),
                    kwargs.get("gene_context", ""),
                )
            elif prompt_name == "grok_enhancement":
                return self.format_grok_enhancement_prompt(
                    kwargs.get("base_response", ""),
                    kwargs.get("original_prompt", ""),
                    kwargs.get("gene_names", []),
                )
            elif prompt_name == "gene_summary":
                return self.format_gene_summary_prompt(kwargs.get("gene_name", ""))
            elif prompt_name == "gene_identification":
                return self.format_gene_identification_prompt(kwargs.get("sequence", ""))
            elif prompt_name == "rag_general":
                return self.format_rag_general_prompt(kwargs.get("user_question", ""))
            elif prompt_name == "rag_gene_specific":
                return self.format_rag_gene_specific_prompt(
                    kwargs.get("gene_name", ""), kwargs.get("user_question", "")
                )
            elif prompt_name == "gene_context":
                return self.format_gene_context(
                    kwargs.get("gene_names", []), kwargs.get("detailed_gene_info", "")
                )
            elif prompt_name == "error_response":
                return self.format_error_response(kwargs.get("gene_names", []))
            elif prompt_name == "personalized_note":
                return self.format_personalized_note(kwargs.get("gene_names", []))
            elif prompt_name == "rag_personalized_note":
                return self.format_rag_personalized_note(kwargs.get("gene_names", []))
            elif prompt_name == "pinecone_results":
                return self.format_pinecone_results_prompt(
                    kwargs.get("context", ""), kwargs.get("user_message", "")
                )
            elif prompt_name == "gene_identification_summary":
                return self.format_gene_identification_summary_prompt(
                    kwargs.get("database_search_results", "")
                )
            elif prompt_name == "gene_agent":
                return self.format_gene_agent_prompt()
            elif prompt_name == "gene_insights":
                return self.format_gene_insights_prompt(**kwargs)
            elif prompt_name == "genome_insights":
                return self.format_genome_insights_prompt(**kwargs)
            elif prompt_name == "variant_analysis":
                return self.format_variant_analysis_prompt(**kwargs)
            else:
                raise ValueError(f"Unknown prompt name: {prompt_name}")

        except Exception as e:
            logger.error(f"Error formatting prompt '{prompt_name}': {e}")
            raise

    def format_gene_insights_prompt(self, analysis_data: str = "") -> str:
        return PromptTemplates.get_gene_insights_prompt().format(analysis_data=analysis_data)

    def format_genome_insights_prompt(self, genome_summary: str = "") -> str:
        return PromptTemplates.get_genome_insights_prompt().format(genome_summary=genome_summary)

    def format_variant_analysis_prompt(self, annotation_data: str = "") -> str:
        return PromptTemplates.get_variant_analysis_prompt().format(annotation_data=annotation_data)

    def format_gene_agent_complete_analysis_prompt(self, sequence: str = "") -> str:
        return PromptTemplates.get_gene_agent_complete_analysis_prompt().format(sequence=sequence[:200] + ('...' if len(sequence) > 200 else ''))

    def get_available_prompts(self) -> list[str]:
        return [
            "grok_general",
            "grok_enhancement",
            "gene_summary",
            "gene_identification",
            "rag_general",
            "rag_gene_specific",
            "gene_context",
            "error_response",
            "personalized_note",
            "rag_personalized_note",
            "pinecone_results",
            "gene_identification_summary",
            "gene_agent",
            "gene_insights",
            "genome_insights",
            "variant_analysis",
        ]

    def clear_cache(self):
        self._prompt_cache.clear()
        logger.info("Prompt cache cleared")

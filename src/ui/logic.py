import logging
from typing import Any, Optional

from ..constants.constants import *
from ..settings import settings

from ..core.chat_manager import ChatManager
from ..core.gene_agent import GeneAgent
from ..core.gene_extractor import GeneExtractor
from ..core.summary_generator import SummaryGenerator
from ..models.app_models import (
    AnalysisResult,
    AppGenomeProcessResult,
    ComponentStatus,
    ComponentStatusInfo,
    Document,
    WorkflowStatus,
)
from ..models.rag_models import RagChatResult
from ..rag.genome_rag import GenomeRAG

logger = logging.getLogger(__name__)

GENE_EXTRACTION_KEYS = ["genes_found", "genes", "gene_names"]


class AppLogic:
    def __init__(self) -> None:
        self.gene_agent: Optional[GeneAgent] = None
        self.genome_rag: Optional[GenomeRAG] = None
        self.gene_extractor = GeneExtractor()
        self.summary_generator = SummaryGenerator()
        self.chat_manager: Optional[ChatManager] = None
        self.settings = settings

    def analyze_sequence(self, sequence: str) -> AnalysisResult:
        try:
            return self._analyze_with_traditional_agent(sequence)

        except Exception as e:
            logger.error(f"Sequence analysis failed: {e}")
            return AnalysisResult(
                success=False,
                sequence_length=len(sequence),
                results={},
                method=UNKNOWN_LABEL,
                message="Analysis failed",
                error=f"Analysis failed: {str(e)}",
            )

    def _analyze_with_traditional_agent(self, sequence: str) -> AnalysisResult:
        if not self.gene_agent:
            self.gene_agent = GeneAgent()

        result = self.gene_agent.identify_genes(sequence)

        if result["success"]:
            results_data = self._extract_traditional_results(result)

            self._store_results_in_pinecone(sequence, results_data)

            return AnalysisResult(
                success=True,
                sequence_length=len(sequence),
                results=results_data,
                method=result.get("method", "LangChain ReAct Agent"),
                message="Genes identified successfully",
            )
        else:
            return AnalysisResult(
                success=False,
                sequence_length=len(sequence),
                results={},
                method="LangChain ReAct Agent",
                message="Analysis failed",
                error=result["error"],
            )

    def _extract_traditional_results(self, result: dict[str, Any]) -> dict[str, Any]:
        return result.get("agent_output") or result.get("tool_result") or result

    def _store_results_in_pinecone(self, sequence: str, results_data: dict[str, Any]) -> None:
        try:
            if not self.genome_rag:
                self._setup_genome_rag()

            if not self.genome_rag or not self.genome_rag.vector_store:
                logger.warning("Genome RAG not available for storage")
                return

            documents = self._create_documents_from_results(sequence, results_data)

            if documents:
                document_dicts = [
                    {"content": doc.content, "metadata": doc.metadata} for doc in documents
                ]
                success = self.genome_rag.vector_store.add_documents(document_dicts)
                if success:
                    logger.info(f"Stored {len(documents)} documents in Pinecone")

        except Exception as e:
            logger.error(f"Failed to store results in Pinecone: {e}")

    def _create_documents_from_results(
        self, sequence: str, results_data: dict[str, Any]
    ) -> list[Document]:
        documents = []

        documents.append(
            Document(
                content=f"DNA sequence analysis: {len(sequence)} base pairs analyzed. Found genes through bioinformatics analysis.",
                metadata={"type": "sequence_analysis", "sequence_length": len(sequence)},
            )
        )

        genes = self._extract_genes_from_results(results_data)
        if genes:
            documents.append(
                Document(
                    content=f"Genes identified: {', '.join(genes[:self.settings.default_top_genes_limit])}. These genes were found through sequence analysis.",
                    metadata={
                        "type": "gene_identification",
                        "genes": genes[:self.settings.default_top_genes_limit],
                        "gene_count": len(genes),
                    },
                )
            )

        return documents

    def _extract_genes_from_results(self, results_data: dict[str, Any]) -> list[str]:
        try:
            return self.gene_extractor.extract_gene_names_from_results(results_data)
        except Exception:
            genes: list[str] = []
            if isinstance(results_data, dict):
                for key in GENE_EXTRACTION_KEYS:
                    if key in results_data and isinstance(results_data[key], list):
                        genes.extend(results_data[key])
            return list(set(genes))

    def _setup_genome_rag(self) -> None:
        try:
            config = self._build_rag_config()
            self.genome_rag = GenomeRAG(config)
        except Exception as e:
            logger.error(f"Failed to setup genome RAG: {e}")

    def _build_rag_config(self) -> dict[str, Any]:
        return {
            "embedding_model": self.settings.embedding_model,
            "xai_api_key": self.settings.xai_api_key,
            "collection_name": self.settings.collection_name,
            "pinecone_api_key": self.settings.pinecone_api_key,
            "annotation_requests_per_second": self.settings.annotation_requests_per_second,
            "max_concurrent_requests": self.settings.annotation_max_concurrent_requests,
            "annotation_max_retries": self.settings.annotation_max_retries,
            "enable_annotation_cache": True,
            "annotation_cache_ttl": ANNOTATION_CACHE_TTL,
        }

    def process_genome_file(
        self, file_path: str, file_type: str = "auto"
    ) -> AppGenomeProcessResult:
        if not self.genome_rag:
            self._setup_genome_rag()

        if not self.genome_rag:
            return AppGenomeProcessResult(success=False, error="Genome RAG system not available")

        try:
            result = self.genome_rag.process_genome_file(file_path, file_type)

            if result.success:
                index_result = self.genome_rag.build_index(result.documents)

                summary = self.summary_generator.generate_genome_summary(result.__dict__)

                return AppGenomeProcessResult(
                    success=True,
                    file_type=result.file_type,
                    parsed_count=result.parsed_count,
                    annotated_count=result.annotated_count,
                    documents_count=result.documents_count,
                    index_built=index_result.success,
                    index_error=index_result.error if not index_result.success else None,
                    summary=summary,
                )
            else:
                return AppGenomeProcessResult(success=False, error=result.error)

        except Exception as e:
            logger.error(f"Failed to process genome file {file_path}: {e}")
            return AppGenomeProcessResult(success=False, error=str(e))

    def chat_with_genome(self, query: str) -> str:
        if not self.genome_rag:
            return "Genome RAG system not available. Please process a genome file first."

        try:
            result = self.genome_rag.chat(query)
            if isinstance(result, RagChatResult):
                return result.text or "No relevant genome data found for your query."
            return str(result)
        except Exception as e:
            logger.error(f"Genome chat failed: {e}")
            return f"Error processing your question: {str(e)}"

    def get_chat_response(self, prompt: str, gene_names: list[str]) -> str:
        if not self.chat_manager:
            self.chat_manager = ChatManager(rag_system=self.genome_rag)

        conversation_history = self._get_conversation_history()

        try:
            response = self.chat_manager.get_chat_response(
                prompt,
                gene_names,
                conversation_history,
                strict_mode=self.settings.default_strict_mode,
                min_relevance=CHAT_DEFAULT_MIN_RELEVANCE,
            )

            return response
        except Exception as e:
            logger.error(f"Chat response failed: {e}")
            return f"Chat failed: {str(e)}"

    def _get_conversation_history(self) -> Optional[list[dict[str, str]]]:
        try:
            import streamlit as st

            if hasattr(st, "session_state") and "chat_history" in st.session_state:
                return st.session_state.chat_history
        except Exception:
            pass
        return None

    def extract_gene_names_from_results(self, results: dict[str, Any]) -> list[str]:
        return self.gene_extractor.extract_gene_names_from_results(results)

    def generate_summary(
        self,
        sequence: str,
        gene_names: list[str],
        analysis_results: dict[str, Any],
        rag_results: dict[str, Any],
    ) -> str:
        summary = self.summary_generator.generate_summary(
            sequence, gene_names, analysis_results, rag_results
        )
        try:
            if summary and summary.strip():
                if not self.genome_rag:
                    self._setup_genome_rag()
                if self.genome_rag and self.genome_rag.vector_store:
                    document = Document(
                        content=summary.strip(),
                        metadata={
                            "type": "analysis_summary",
                            "gene_count": len(gene_names),
                            "sequence_length": len(sequence),
                        },
                    )
                    self.genome_rag.vector_store.add_documents(
                        [{"content": document.content, "metadata": document.metadata}]
                    )
        except Exception as e:
            logger.warning(f"Failed to store analysis summary in vector store: {e}")
        return summary

    def get_workflow_status(self) -> WorkflowStatus:
        return WorkflowStatus(
            traditional_gene_agent=self._get_traditional_agent_status(),
            genome_rag=self._get_genome_rag_status(),
        )

    def _get_traditional_agent_status(self) -> Optional[ComponentStatusInfo]:
        try:
            if self.gene_agent:
                return ComponentStatusInfo(
                    agent_type="LangChain ReAct Agent",
                    status=ComponentStatus.INITIALIZED,
                    details={
                        "tools_available": len(self.gene_agent.tools),
                        "status": "initialized",
                    },
                )
        except Exception as e:
            logger.warning(f"Could not get traditional gene agent status: {e}")
        return None

    def _get_genome_rag_status(self) -> Optional[ComponentStatusInfo]:
        try:
            if self.genome_rag:
                return ComponentStatusInfo(
                    agent_type="Genome RAG System",
                    status=ComponentStatus.INITIALIZED,
                    details={
                        "status": "initialized",
                        "vector_store_available": self.genome_rag.vector_store is not None,
                    },
                )
        except Exception as e:
            logger.warning(f"Could not get genome RAG status: {e}")
        return None

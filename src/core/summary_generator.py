import logging
from typing import Any

from .llm_factory import create_llm
from ..prompts.prompt_manager import PromptManager
from . import summary_templates as templates
from ..constants.constants import *
from ..settings import settings

logger = logging.getLogger(__name__)


class SummaryGenerator:
    def __init__(self):
        self.prompt_manager = PromptManager()
        self.llm = create_llm()

    def generate_summary(
        self,
        sequence: str,
        gene_names: list[str],
        analysis_results: dict[str, Any],
        rag_results: dict[str, Any],
    ) -> str:
        try:
            summary_parts = []

            if analysis_results.get("success") and gene_names:
                insights = self._generate_gene_insights(gene_names, analysis_results)
                if insights:
                    summary_parts.append(insights)

            return "\n".join(summary_parts)

        except Exception as e:
            logger.error(f"Enhanced summary generation failed: {e}")
            return f"Summary generation failed: {str(e)}"

    def generate_genome_summary(self, genome_results: dict[str, Any]) -> str:
        try:
            if not genome_results.get("success"):
                return templates.GENOME_PROCESSING_FAILED_TEMPLATE.format(
                    error=genome_results.get("error", UNKNOWN_ERROR)
                )

            summary_parts = []

            summary_parts.append(templates.GENOME_PROCESSING_HEADER)
            summary_parts.append(f"**File Type**: {genome_results.get('file_type', UNKNOWN_LABEL)}")
            summary_parts.append(f"**Items Parsed**: {genome_results.get('parsed_count', 0)}")
            summary_parts.append(
                f"**Documents Created**: {genome_results.get('documents_count', 0)}"
            )

            if genome_results.get("index_built"):
                summary_parts.append(f"**Vector Index**: {templates.VECTOR_INDEX_SUCCESS}")
            else:
                summary_parts.append(f"**Vector Index**: {templates.VECTOR_INDEX_FAILED}")

            if genome_results.get("data"):
                insights = self._generate_genome_insights(genome_results["data"])
                if insights:
                    summary_parts.append(insights)

            return "\n".join(summary_parts)

        except Exception as e:
            logger.error(f"Genome summary generation failed: {e}")
            return templates.GENOME_SUMMARY_FAILED_TEMPLATE.format(error=str(e))

    def _generate_gene_insights(
        self, gene_names: list[str], analysis_results: dict[str, Any]
    ) -> str:
        try:
            if not gene_names:
                return templates.NO_GENES_MESSAGE

            if not self.llm:
                return templates.NO_LLM_GENE_INSIGHTS_MESSAGE

            gene_details = []
            if analysis_results.get("database_results"):
                db_results = analysis_results["database_results"]
                for gene_name in gene_names[:MAX_GENES_FOR_INSIGHTS]:
                    gene_info = self._extract_gene_info(gene_name, db_results, gene_names)
                    if gene_info:
                        gene_details.append(gene_info)

            analysis_data = templates.GENE_ANALYSIS_DATA_TEMPLATE.format(
                gene_names=", ".join(gene_names[:settings.max_genes_for_display]),
                gene_details=(
                    chr(10).join(gene_details[:MAX_GENE_DETAILS])
                    if gene_details
                    else templates.NO_DETAILED_GENE_INFO
                ),
            )

            prompt = self.prompt_manager.format_gene_insights_prompt(analysis_data=analysis_data)

            response = self.llm.invoke(prompt)
            insights = self._extract_clean_text(response)

            if insights and len(insights) > MIN_INSIGHTS_LENGTH:
                return insights
            else:
                return templates.INSUFFICIENT_GENE_INSIGHTS

        except Exception as e:
            logger.warning(f"Failed to generate gene insights: {e}")
            return templates.INSUFFICIENT_GENE_INSIGHTS

    def _generate_genome_insights(self, genome_data: list[dict[str, Any]]) -> str:
        try:
            if not genome_data:
                return templates.NO_GENOME_DATA_MESSAGE

            if not self.llm:
                return templates.NO_LLM_GENOME_INSIGHTS_MESSAGE

            sequence_count = sum(1 for item in genome_data if "sequence" in item)
            variant_count = sum(1 for item in genome_data if "chrom" in item)
            annotated_count = sum(1 for item in genome_data if "annotation" in item)

            sample_annotations = []
            for item in genome_data[:MAX_GENOME_SAMPLE_ITEMS]:
                if "annotation" in item:
                    annotation = item["annotation"]
                    if "sources" in annotation:
                        sources = []
                        for source, data in annotation["sources"].items():
                            if isinstance(data, dict) and data.get("status") == "ok":
                                sources.append(source)
                        if sources:
                            sample_annotations.append(
                                f"- {item.get('id', UNKNOWN_LABEL)}: {', '.join(sources)}"
                            )

            genome_summary = templates.GENOME_DATA_SUMMARY_TEMPLATE.format(
                sequence_count=sequence_count,
                variant_count=variant_count,
                annotated_count=annotated_count,
                sample_annotations=(
                    chr(10).join(sample_annotations)
                    if sample_annotations
                    else templates.NO_ANNOTATIONS_AVAILABLE
                ),
            )

            prompt = self.prompt_manager.format_genome_insights_prompt(
                genome_summary=genome_summary
            )

            response = self.llm.invoke(prompt)
            insights = self._extract_clean_text(response)

            if insights and len(insights) > MIN_INSIGHTS_LENGTH:
                return insights
            else:
                return templates.INSUFFICIENT_GENOME_INSIGHTS

        except Exception as e:
            logger.warning(f"Failed to generate genome insights: {e}")
            return templates.INSUFFICIENT_GENOME_INSIGHTS

    def _extract_gene_info(self, gene_name: str, db_results: dict[str, Any], extracted_genes: list[str] = None) -> str:
        try:
            if extracted_genes and gene_name not in extracted_genes:
                return ""
            
            gene_info_parts = [f"Gene: {gene_name}"]

            if "uniprot_results" in db_results:
                for hit in db_results["uniprot_results"]:
                    # Check if this hit was the source of our extracted gene name
                    hit_gene_name = hit.get("gene_name", "")
                    if hit_gene_name and hit_gene_name.upper() == gene_name.upper():
                        gene_info_parts.append(f" - Protein: {hit.get('protein_name', UNKNOWN_LABEL)}")
                        gene_info_parts.append(
                            f" - Function: {hit.get('function', UNKNOWN_LABEL)[:MAX_FUNCTION_DESCRIPTION_LENGTH]}..."
                        )
                        break
                    elif isinstance(hit.get("gene_names"), list):
                        for gn in hit.get("gene_names", []):
                            if gn.upper() == gene_name.upper():
                                gene_info_parts.append(f"  - Protein: {hit.get('protein_name', UNKNOWN_LABEL)}")
                                gene_info_parts.append(
                                    f" - Function: {hit.get('function', UNKNOWN_LABEL)[:MAX_FUNCTION_DESCRIPTION_LENGTH]}..."
                                )
                                break
                        if len(gene_info_parts) > 1: 
                            break

            if "ncbi_protein_results" in db_results:
                for hit in db_results["ncbi_protein_results"]:
                    hit_gene_name = hit.get("gene_name", "")
                    if hit_gene_name and hit_gene_name.upper() == gene_name.upper():
                        gene_info_parts.append(
                            f"  - Description: {hit.get('description', UNKNOWN_LABEL)[:MAX_FUNCTION_DESCRIPTION_LENGTH]}..."
                        )
                        break

            return "\n".join(gene_info_parts) if len(gene_info_parts) > 1 else ""

        except Exception as e:
            logger.warning(f"Failed to extract gene info for {gene_name}: {e}")
            return templates.NO_GENE_INFO_MESSAGE.format(gene_name=gene_name)

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
            logger.warning(f"Error extracting clean text from response: {e}")
            return str(response)

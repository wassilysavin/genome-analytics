import logging
from typing import Any
from ...constants.constants import *

from langchain.tools import tool

from ...models.gene_identification_models import (
    AnalyzedORF,
    DatabaseSearchResults,
    SequenceAnalysis,
)
from ..bio.bio_analysis import (
    analyze_protein_properties,
    calculate_sequence_composition,
    find_orfs,
    translate_orf,
)
from .blast_local_provider import BlastLocalProvider
from .base_search import DatabaseType, SearchType

logger = logging.getLogger(__name__)


@tool
def analyze_dna_sequence(
    sequence: str, min_orf_length: int = BIO_MIN_ORF_LENGTH
) -> dict[str, Any]:
    """
    Analyze DNA sequence for ORFs and basic composition.

    Args:
        sequence: DNA sequence to analyze
        min_orf_length: Minimum ORF length in base pairs

    Returns:
        Dictionary with ORF analysis and sequence composition
    """
    try:
        analysis = _perform_sequence_analysis(sequence, min_orf_length)

        return _sequence_analysis_to_dict(analysis)

    except Exception as e:
        logger.error(f"DNA sequence analysis failed: {e}")
        return {"success": False, "error": str(e)}


def _perform_sequence_analysis(sequence: str, min_orf_length: int) -> SequenceAnalysis:
    composition = calculate_sequence_composition(sequence)

    orfs = find_orfs(sequence, min_orf_length)

    analyzed_orfs = _analyze_top_orfs(orfs)

    return SequenceAnalysis(
        success=True,
        sequence_composition=composition,
        total_orfs_found=len(orfs),
        analyzed_orfs=analyzed_orfs,
        analysis_method="BioPython ORF detection with protein analysis",
    )


def _analyze_top_orfs(orfs: list[dict[str, Any]]) -> list[AnalyzedORF]:
    analyzed_orfs = []

    for i, orf in enumerate(orfs[:MAX_ORFS_TO_ANALYZE]):
        try:
            protein_seq = translate_orf(orf["sequence"])
            if protein_seq and len(protein_seq) > BIO_MIN_PROTEIN_LENGTH:
                protein_props = analyze_protein_properties(protein_seq)

                analyzed_orf = AnalyzedORF(
                    orf_id=str(i + 1),
                    genomic_position={
                        "start": orf["start"],
                        "end": orf["end"],
                        "strand": orf["strand"],
                        "frame": orf["frame"],
                    },
                    length_bp=orf["length"],
                    protein_sequence=protein_seq,
                    protein_properties=protein_props,
                    frame=orf["frame"],
                )
                analyzed_orfs.append(analyzed_orf)

        except Exception as e:
            logger.warning(f"Failed to analyze ORF {i+1}: {e}")
            continue

    return analyzed_orfs


def _sequence_analysis_to_dict(analysis: SequenceAnalysis) -> dict[str, Any]:
    return {
        "success": analysis.success,
        "sequence_composition": analysis.sequence_composition,
        "total_orfs_found": analysis.total_orfs_found,
        "analyzed_orfs": [_analyzed_orf_to_dict(orf) for orf in analysis.analyzed_orfs],
        "analysis_method": analysis.analysis_method,
        "error": analysis.error,
    }


def _analyzed_orf_to_dict(orf: AnalyzedORF) -> dict[str, Any]:
    return {
        "orf_id": orf.orf_id,
        "genomic_position": orf.genomic_position,
        "length_bp": orf.length_bp,
        "protein_sequence": orf.protein_sequence,
        "protein_properties": orf.protein_properties,
        "frame": orf.frame,
    }


def _search_databases_for_genes_internal(
    sequence: str, min_orf_length: int = BIO_MIN_ORF_LENGTH
) -> dict[str, Any]:
    """
    A function to search biological databases (NCBI, UniProt) for gene identification.

    Args:
        sequence: DNA sequence to analyze
        min_orf_length: Minimum ORF length in base pairs

    Returns:
        Dictionary with database search results
    """
    try:
        sequence_analysis = _perform_sequence_analysis(sequence, min_orf_length)

        if not sequence_analysis.success:
            return _sequence_analysis_to_dict(sequence_analysis)

        database_results = _search_databases_for_genes(sequence_analysis.analyzed_orfs)

        return _combine_analysis_results(sequence_analysis, database_results)

    except Exception as e:
        logger.error(f"Database search failed: {e}")
        return {"success": False, "error": str(e)}


def _search_databases_for_genes(analyzed_orfs: list[AnalyzedORF]) -> DatabaseSearchResults:
    database_results = DatabaseSearchResults(
        ncbi_protein_results=[], ncbi_nucleotide_results=[], uniprot_results=[]
    )

    logger.info(f"Found {len(analyzed_orfs)} ORFs to analyze")
    for orf in analyzed_orfs[:MAX_ORFS_FOR_DATABASE_SEARCH]:
        _search_orf_in_databases(orf, database_results)

    database_results.total_matches = (
        len(database_results.ncbi_protein_results)
        + len(database_results.ncbi_nucleotide_results)
        + len(database_results.uniprot_results)
    )

    return database_results


def _search_orf_in_databases(orf: AnalyzedORF, database_results: DatabaseSearchResults) -> None:
    protein_seq = orf.protein_sequence
    logger.info(f"Analyzing ORF {orf.orf_id} with protein sequence: {protein_seq[:20]}...")

    _search_ncbi_protein_database(orf, protein_seq, database_results)


def _search_ncbi_protein_database(
    orf: AnalyzedORF, protein_seq: str, database_results: DatabaseSearchResults
) -> None:
    try:
        provider = BlastLocalProvider(DatabaseType.BLAST_LOCAL, SearchType.PROTEIN)
        if not provider.setup():
            logger.warning("Failed to setup local BLAST provider")
            return
        search_result = provider.search(protein_seq, max_results=3)
        ncbi_protein_result = search_result.to_dict()
        logger.info(
            f"BLAST result for {orf.orf_id}: found_matches={ncbi_protein_result.get('found_matches')}, hits={len(ncbi_protein_result.get('hits', []))}"
        )

        if ncbi_protein_result.get("found_matches"):
            for hit in ncbi_protein_result["hits"]:
                hit["source_orf"] = orf.orf_id
                hit["orf_position"] = orf.genomic_position
                logger.info(
                    f"  Hit: {hit.get('gene_name', UNKNOWN_LABEL)} - {hit.get('description', NO_DESCRIPTION)[:LOG_DESC_PREVIEW_SHORT]}..."
                )
            database_results.ncbi_protein_results.extend(ncbi_protein_result["hits"])
        else:
            logger.warning(f"No BLAST matches found for {orf.orf_id}")

    except Exception as e:
        logger.warning(f"Local BLAST search failed for {orf.orf_id}: {e}")


def _combine_analysis_results(
    sequence_analysis: SequenceAnalysis, database_results: DatabaseSearchResults
) -> dict[str, Any]:
    return {
        "success": True,
        "sequence_analysis": _sequence_analysis_to_dict(sequence_analysis),
        "database_results": _database_search_results_to_dict(database_results),
        "total_database_matches": database_results.total_matches,
        "search_method": "Local BLAST against NCBI Swiss-Prot + UniProt REST API",
    }


def _database_search_results_to_dict(database_results: DatabaseSearchResults) -> dict[str, Any]:
    return {
        "ncbi_protein_results": database_results.ncbi_protein_results,
        "ncbi_nucleotide_results": database_results.ncbi_nucleotide_results,
        "uniprot_results": database_results.uniprot_results,
    }


@tool
def search_databases_for_genes(
    sequence: str, min_orf_length: int = BIO_MIN_ORF_LENGTH
) -> dict[str, Any]:
    """
    Search biological databases (NCBI, UniProt) for gene identification.

    Args:
        sequence: DNA sequence to analyze
        min_orf_length: Minimum ORF length in base pairs

    Returns:
        Dictionary with database search results
    """
    return _search_databases_for_genes_internal(sequence, min_orf_length)


@tool
def generate_gene_summary(database_search_results: str) -> dict[str, Any]:
    """
    Generate summary of gene identification results.

    Args:
        database_search_results: String representation of database search results

    Returns:
        Dictionary with a generated summary
    """
    try:
        from ...core.llm_factory import create_llm
        from ...prompts.prompt_manager import PromptManager

        llm = create_llm()
        prompt_manager = PromptManager()

        prompt = prompt_manager.format_gene_identification_summary_prompt(database_search_results)

        response = llm.invoke(prompt)

        if hasattr(response, "content"):
            summary_text = str(response.content)
        elif isinstance(response, dict):
            content_val = response.get("content", "") or response.get("text", "")
            summary_text = str(content_val)
        else:
            summary_text = str(response)

        return {"success": True, "summary": summary_text, "llm_model": "Grok (xAI)"}

    except Exception as e:
        logger.error(f"AI summary generation failed: {e}")
        return {"success": False, "error": f"Summary generation failed: {str(e)}"}


@tool
def generate_detailed_insights(
    database_search_results: str, gene_names: list[str]
) -> dict[str, Any]:
    """
    Generate detailed insights about identified genes including functional categories,
    health implications, biological connections, and notable characteristics.

    Args:
        database_search_results: String representation of database search results
        gene_names: List of identified gene names

    Returns:
        Dictionary with detailed insights
    """
    try:
        from ...core.llm_factory import create_llm
        from ...prompts.prompt_manager import PromptManager

        llm = create_llm()
        prompt_manager = PromptManager()

        insights_prompt = prompt_manager.get_prompt(
            "gene_detailed_insights",
            database_search_results=database_search_results,
            gene_names=gene_names,
        )

        response = llm.invoke(insights_prompt)

        if hasattr(response, "content"):
            insights_text = str(response.content)
        elif isinstance(response, dict):
            content_val = response.get("content", "") or response.get("text", "")
            insights_text = str(content_val)
        else:
            insights_text = str(response)

        return {"success": True, "insights": insights_text, "llm_model": "Grok (xAI)"}

    except Exception as e:
        logger.error(f"Detailed insights generation failed: {e}")
        return {"success": False, "error": f"Insights generation failed: {str(e)}"}


@tool
def identify_genes_complete(
    sequence: str, min_orf_length: int = BIO_MIN_ORF_LENGTH
) -> dict[str, Any]:
    """
    Complete gene identification workflow: analyze sequence, search databases, generate summary.

    Args:
        sequence: DNA sequence to analyze
        min_orf_length: Minimum ORF length in base pairs

    Returns:
        Dictionary with complete gene identification results
    """
    try:
        # Step 1: Search databases
        database_results = search_databases_for_genes.invoke(
            {"sequence": sequence, "min_orf_length": min_orf_length}
        )

        if not database_results.get("success"):
            return database_results

        # Step 2: Generate summary
        summary_results = generate_gene_summary.invoke(
            {"database_search_results": str(database_results)}
        )

        # Step 3: Generate detailed insights
        gene_names = []
        if "genes_found" in database_results:
            gene_names = database_results["genes_found"]
        elif "genes" in database_results:
            gene_names = database_results["genes"]

        detailed_insights = None
        if gene_names:
            detailed_insights = generate_detailed_insights.invoke(
                {"database_search_results": str(database_results), "gene_names": gene_names}
            )

        # Step 4: Combine results
        final_result = {
            "success": True,
            "sequence_length": len(sequence),
            "analysis_method": "Complete workflow: BioPython + NCBI + UniProt + AI",
            "sequence_analysis": database_results["sequence_analysis"],
            "database_search": database_results["database_results"],
            "ai_summary": summary_results,
            "detailed_insights": detailed_insights,
            "total_database_matches": database_results["total_database_matches"],
            "message": "Complete gene identification workflow completed successfully",
        }

        return final_result

    except Exception as e:
        logger.error(f"Complete gene identification failed: {e}")
        return {"success": False, "error": str(e)}

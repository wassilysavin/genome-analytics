NO_GENES_MESSAGE = "No genes identified for analysis."
NO_LLM_GENE_INSIGHTS_MESSAGE = "LLM not available for generating insights."
NO_GENOME_DATA_MESSAGE = "No genome data available for analysis."
NO_LLM_GENOME_INSIGHTS_MESSAGE = "LLM not available for generating genome insights."
NO_GENE_INFO_MESSAGE = "No detailed information available for {gene_name}"
INSUFFICIENT_GENE_INSIGHTS = "Insufficient data to generate gene insights."
INSUFFICIENT_GENOME_INSIGHTS = "Insufficient data to generate genome insights."

GENOME_FILE_SUMMARY_TEMPLATE = """## Genome File Processing Summary
**File Type**: {file_type}
**Items Parsed**: {parsed_count}
**Documents Created**: {documents_count}
**Vector Index**: {index_status}"""

VECTOR_INDEX_SUCCESS = "Successfully built"
VECTOR_INDEX_FAILED = "Failed to build"

GENE_ANALYSIS_DATA_TEMPLATE = """Genes found: {gene_names}

Gene details:
{gene_details}"""

GENOME_DATA_SUMMARY_TEMPLATE = """Genome data summary:
- Sequences: {sequence_count}
- Variants: {variant_count}
- Annotated items: {annotated_count}

Sample annotations:
{sample_annotations}"""

NO_ANNOTATIONS_AVAILABLE = "No annotations available"
NO_DETAILED_GENE_INFO = "No detailed gene information available"

GENE_INFO_TEMPLATE = """Gene: {gene_name}
  - Protein: {protein_name}
  - Function: {function}"""

GENE_INFO_NCBI_TEMPLATE = """Gene: {gene_name}
  - Description: {description}"""

FALLBACK_HIGH_DENSITY_INSIGHT = (
    "**High gene density**: Found a large number of genes, suggesting a complex genetic profile"
)
FALLBACK_CLINICAL_INSIGHT = (
    "**Clinically significant genes**: Contains well-studied genes with known health implications"
)
FALLBACK_DIVERSITY_INSIGHT = "**Genetic diversity**: Identified {gene_count} distinct genes across multiple functional categories"
FALLBACK_GENE_DEFAULT = (
    "Analysis completed successfully - detailed insights available through chat interface"
)

FALLBACK_SEQUENCE_INSIGHT = "**Sequence data**: {sequence_count} sequences processed for analysis"
FALLBACK_VARIANT_INSIGHT = "**Variant data**: {variant_count} variants identified for annotation"
FALLBACK_MIXED_DATA_INSIGHT = (
    "**Mixed data types**: Contains both sequence and variant information"
)
FALLBACK_GENOME_DEFAULT = (
    "Genome processing completed - detailed insights available through chat interface"
)

GENOME_PROCESSING_FAILED_TEMPLATE = "Genome processing failed: {error}"
GENOME_SUMMARY_FAILED_TEMPLATE = "Genome summary generation failed: {error}"

# Section headers
GENOME_PROCESSING_HEADER = "## Genome File Processing Summary"
KEY_INSIGHTS_HEADER = "## Key Insights"
GENE_NAMES_HEADER = "**Gene Names**: {gene_names}"
GENES_IDENTIFIED_HEADER = "**Genes Identified**: {count} genes found"
SEQUENCE_HEADER = "**Sequence**: {length:,} base pairs analyzed"

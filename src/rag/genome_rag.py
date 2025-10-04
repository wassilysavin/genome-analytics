import logging
from pathlib import Path
from typing import Any
from ..constants.constants import *
from ..settings import settings

from ..core.llm_factory import create_llm
from ..models.annotation_models import VariantInfo, AnnotationConfig
from ..models.rag_models import (
    DocumentType,
    FileType,
    GenomeProcessResult,
    IndexBuildResult,
    ProcessedDocument,
    RagChatResult,
    RAGConfig,
    SearchResult,
    VectorStoreConfig,
)
from ..prompts.prompt_manager import PromptManager
from .embedding_providers import create_embedding_provider
from .vector_store import create_vector_store


class GenomeParser:
    def parse_fasta(self, content: str) -> list[dict[str, Any]]:
        sequences = []
        current_seq = ""
        current_header = ""

        lines = content.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if current_header and current_seq:
                    sequences.append({
                        'header': current_header,
                        'sequence': current_seq,
                        'type': 'fasta'
                    })
                current_header = line[1:]  # Remove '>'
                current_seq = ""
            else:
                current_seq += line.upper()

        # Save last sequence
        if current_header and current_seq:
            sequences.append({
                'header': current_header,
                'sequence': current_seq,
                'type': 'fasta'
            })

        return sequences

    def parse_fastq(self, content: str) -> list[dict[str, Any]]:
        # basic implementation for now
        lines = content.strip().split('\n')
        sequences = []

        for i in range(0, len(lines), 4):
            if i + 3 < len(lines):
                header = lines[i][1:] if lines[i].startswith('@') else lines[i]
                sequence = lines[i + 1].upper()
                quality_header = lines[i + 2]
                quality = lines[i + 3]

                sequences.append({
                    'header': header,
                    'sequence': sequence,
                    'quality': quality,
                    'type': 'fastq'
                })

        return sequences

    def parse_vcf(self, content: str) -> list[dict[str, Any]]:
        lines = content.strip().split('\n')
        variants = []

        for line in lines:
            if not line.startswith('#') and line.strip():
                fields = line.strip().split('\t')
                if len(fields) >= 8:
                    variants.append({
                        'chrom': fields[0],
                        'pos': fields[1],
                        'id': fields[2],
                        'ref': fields[3],
                        'alt': fields[4],
                        'qual': fields[5],
                        'filter': fields[6],
                        'info': fields[7],
                        'type': 'vcf'
                    })

        return variants


logger = logging.getLogger(__name__)

FILE_EXTENSION_MAP = {
    ".fa": FileType.FASTA,
    ".fasta": FileType.FASTA,
    ".fq": FileType.FASTQ,
    ".fastq": FileType.FASTQ,
    ".vcf": FileType.VCF,
}

EMBEDDING_DIMENSIONS = {
    "ada-002": EMBEDDING_DIMENSION_ADA_002, 
    "default": EMBEDDING_DIMENSION_DEFAULT
}


class GenomeRAG:
    def __init__(self, config: dict[str, Any]):
        self.config = self._build_rag_config(config)
        self.parser = GenomeParser()
        self.llm = create_llm()
        self.prompt_manager = PromptManager()
        self._initialize_annotator()
        self._initialize_vector_store()
        self.documents: list[ProcessedDocument] = []

    def _build_rag_config(self, config: dict[str, Any]) -> RAGConfig:
        return RAGConfig(
            annotation_requests_per_second=config.get("annotation_requests_per_second", settings.annotation_requests_per_second),
            max_concurrent_requests=config.get("max_concurrent_requests", settings.annotation_max_concurrent_requests),
            annotation_max_retries=config.get("annotation_max_retries", settings.annotation_max_retries),
            enable_annotation_cache=config.get("enable_annotation_cache", True),
            annotation_cache_ttl=config.get("annotation_cache_ttl", ANNOTATION_CACHE_TTL),
            embedding_model=config.get("embedding_model", settings.embedding_model),
            collection_name=config.get("collection_name", settings.collection_name),
            dimension=config.get("dimension", RAG_DEFAULT_DIMENSION),
            ncbi_api_key=config.get("ncbi_api_key"),
            omim_api_key=config.get("omim_api_key"),
            pinecone_api_key=config.get("pinecone_api_key", settings.pinecone_api_key),
        )

    def _initialize_annotator(self) -> None:
        try:
            annotation_config = AnnotationConfig(
                requests_per_second=self.config.annotation_requests_per_second,
                max_concurrent_requests=self.config.max_concurrent_requests,
                max_retries=self.config.annotation_max_retries,
                enable_cache=self.config.enable_annotation_cache,
                cache_ttl=self.config.annotation_cache_ttl,
                ncbi_api_key=self.config.ncbi_api_key,
                omim_api_key=self.config.omim_api_key,
            )
            self.annotator = GenomeAnnotator(annotation_config)
            logger.info("Genome annotator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize annotator: {e}")
            self.annotator = None

    def _initialize_vector_store(self) -> None:
        try:
            embedding_provider = create_embedding_provider(self.config.embedding_model)
            dimension = self._get_embedding_dimension(embedding_provider)
            vector_config = VectorStoreConfig(
                collection_name=self.config.collection_name,
                dimension=dimension,
                pinecone_api_key=self.config.pinecone_api_key,
            )
            self.vector_store = create_vector_store(vector_config, embedding_provider)
            logger.info(
                f"Production vector store initialized with Sentence Transformers ({self.config.embedding_model}) and Pinecone"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.vector_store = None

    def _get_embedding_dimension(self, embedding_provider) -> int:
        try:
            test_embedding = embedding_provider.encode(["test"])
            return test_embedding.shape[1]
        except Exception:
            return EMBEDDING_DIMENSIONS.get(
                self.config.embedding_model, EMBEDDING_DIMENSIONS["default"]
            )

    def process_genome_file(self, file_path: str, file_type: str = "auto") -> GenomeProcessResult:
        try:
            content = self._read_file_content(file_path)
            detected_type = self._detect_file_type(file_path) if file_type == "auto" else file_type
            parsed_data = self._parse_content(content, detected_type)
            annotated_data = self._annotate_data(parsed_data)
            documents = self._create_documents(annotated_data)
            return GenomeProcessResult(
                success=True,
                file_type=detected_type,
                parsed_count=len(parsed_data),
                annotated_count=len(annotated_data),
                documents_count=len(documents),
                data=annotated_data,
                documents=documents,
            )
        except Exception as e:
            logger.error(f"Failed to process genome file {file_path}: {e}")
            return GenomeProcessResult(success=False, error=str(e))

    def _read_file_content(self, file_path: str) -> str:
        with open(file_path) as f:
            return f.read()

    def _detect_file_type(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext in FILE_EXTENSION_MAP:
            return FILE_EXTENSION_MAP[ext].value
        else:
            raise ValueError(f"Cannot detect file type for {file_path}")

    def _parse_content(self, content: str, file_type: str) -> list[dict[str, Any]]:
        if file_type == FileType.FASTA.value:
            return self.parser.parse_fasta(content)
        elif file_type == FileType.FASTQ.value:
            return self.parser.parse_fastq(content)
        elif file_type == FileType.VCF.value:
            return self.parser.parse_vcf(content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _annotate_data(self, parsed_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.annotator:
            logger.warning("Annotator not available, skipping annotation")
            return parsed_data
        variants, variant_indices = self._extract_variants(parsed_data)
        if variants:
            logger.info(f"Annotating {len(variants)} variants using production annotator")
            annotation_results = self.annotator.annotate_variants(variants)
            self._apply_annotations(parsed_data, annotation_results, variant_indices)
        return parsed_data

    def _extract_variants(
        self, parsed_data: list[dict[str, Any]]
    ) -> tuple[list[VariantInfo], list[int]]:
        variants = []
        variant_indices = []
        for i, item in enumerate(parsed_data):
            if self._is_variant_item(item):
                variant = VariantInfo(
                    chrom=item["chrom"],
                    pos=item["pos"],
                    ref=item["ref"],
                    alt=item["alt"],
                    rs_id=item.get("id") if item.get("id", "").startswith("rs") else None,
                )
                variants.append(variant)
                variant_indices.append(i)
        return variants, variant_indices

    def _is_variant_item(self, item: dict[str, Any]) -> bool:
        return all(key in item for key in ["chrom", "pos", "ref", "alt"])

    def _apply_annotations(
        self,
        parsed_data: list[dict[str, Any]],
        annotation_results: list[Any],
        variant_indices: list[int],
    ) -> None:
        for i, result in enumerate(annotation_results):
            if i < len(variant_indices):
                parsed_data[variant_indices[i]]["annotation"] = result.sources

    def _create_documents(self, annotated_data: list[dict[str, Any]]) -> list[ProcessedDocument]:
        documents = []
        for item in annotated_data:
            if "sequence" in item:
                document = self._create_sequence_document(item)
            elif "chrom" in item:
                document = self._create_variant_document(item)
            else:
                continue
            documents.append(document)
        return documents

    def _create_sequence_document(self, item: dict[str, Any]) -> ProcessedDocument:
        content = f"Sequence: {item['id']}\n"
        content += f"Description: {item.get('description', '')}\n"
        content += f"Length: {item.get('length', 0)} bp\n"
        content += "Type: DNA sequence"
        return ProcessedDocument(
            content=content,
            metadata={
                "type": DocumentType.SEQUENCE.value,
                "id": item["id"],
                "length": item.get("length", 0),
            },
        )

    def _create_variant_document(self, item: dict[str, Any]) -> ProcessedDocument:
        content = f"Variant: {item.get('id', UNKNOWN_LABEL)}\n"
        content += f"Position: {item['chrom']}:{item['pos']}\n"
        content += f"Reference: {item['ref']} → Alternative: {item['alt']}\n"
        if "annotation" in item:
            content += self._format_annotation_content(item["annotation"])
        return ProcessedDocument(
            content=content,
            metadata={
                "type": DocumentType.VARIANT.value,
                "chrom": item["chrom"],
                "pos": item["pos"],
                "ref": item["ref"],
                "alt": item["alt"],
            },
        )

    def _format_annotation_content(self, annotation: dict[str, Any]) -> str:
        content = ""
        if "sources" in annotation:
            for source, data in annotation["sources"].items():
                if isinstance(data, dict) and data.get("status") == "ok":
                    content += f"{source.upper()}: Available\n"
                elif isinstance(data, dict) and "error" in data:
                    content += f"{source.upper()}: Error - {data['error']}\n"
        return content

    def build_index(self, documents: list[ProcessedDocument]) -> IndexBuildResult:
        if not self.vector_store:
            return IndexBuildResult(success=False, error="No vector store available")
        try:
            document_dicts = [
                {"content": doc.content, "metadata": doc.metadata} for doc in documents
            ]
            success = self.vector_store.add_documents(document_dicts)
            if success:
                self.documents = documents
                stats = self.vector_store.get_stats()
                return IndexBuildResult(
                    success=True, documents_count=len(documents), vector_store_stats=stats
                )
            else:
                return IndexBuildResult(
                    success=False, error="Failed to add documents to vector store"
                )
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return IndexBuildResult(success=False, error=str(e))

    def search(self, query: str, top_k: int = settings.vector_search_top_k) -> list[SearchResult]:
        if not self.vector_store:
            return []
        try:
            embedding_provider = self.vector_store.embedding_provider
            query_embedding = embedding_provider.encode([query])[0]
            results = self.vector_store.search(query_embedding, top_k)
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append(
                    SearchResult(
                        rank=i + 1,
                        content=result["content"],
                        metadata=result["metadata"],
                        similarity=result["similarity"],
                    )
                )
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def chat(self, query: str, top_k: int = settings.chat_vector_search_top_k) -> RagChatResult:
        try:
            results = self.search(query, top_k)
            if not results:
                return RagChatResult(has_context=False, text="")

            context = self._build_context_from_results(results)
            if not context.strip():
                return RagChatResult(has_context=False, text="")

            prompt = self.prompt_manager.format_pinecone_results_prompt(
                context=context, user_message=query
            )

            llm_response = self.llm.invoke(prompt)
            if isinstance(llm_response, dict):
                content = llm_response.get("content") or ""
                if isinstance(content, str):
                    text = content.strip()
                    return RagChatResult(has_context=bool(text), text=text)
                text = str(content) if content is not None else ""
                return RagChatResult(has_context=bool(text.strip()), text=text.strip())
            if hasattr(llm_response, "content"):
                text = str(llm_response.content).strip()
                return RagChatResult(has_context=bool(text), text=text)
            text = str(llm_response).strip()
            return RagChatResult(has_context=bool(text), text=text)

        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return RagChatResult(has_context=False, text="")

    def _build_context_from_results(self, results: list[SearchResult]) -> str:
        return "\n\n".join(
            [f"Source {r.rank} (Relevance: {r.similarity:.3f}):\n{r.content}" for r in results]
        )


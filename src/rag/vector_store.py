import logging
import time
from abc import ABC, abstractmethod
from typing import Any
from ..constants.constants import *
from ..settings import settings

import numpy as np

from ..models.rag_models import VectorStoreConfig
from .embedding_providers import EmbeddingProvider

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: list[dict[str, Any]]) -> bool:
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = settings.vector_search_top_k) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        pass


class PineconeVectorStore(VectorStore):
    def __init__(self, config: VectorStoreConfig, embedding_provider: EmbeddingProvider):
        self.config = config
        self.embedding_provider = embedding_provider
        self._client = None
        self._index = None
        self._initialize_client()
        self._initialize_index()

    def _initialize_client(self):
        try:
            from pinecone import Pinecone

            if not self.config.pinecone_api_key:
                raise ValueError("pinecone_api_key required for Pinecone store")

            self._client = Pinecone(api_key=self.config.pinecone_api_key)
            logger.info("Pinecone client initialized successfully")

        except ImportError as e:
            raise ImportError("pinecone package required for Pinecone store") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone client: {e}") from e

    def _initialize_index(self):
        try:
            from pinecone import ServerlessSpec

            existing_indexes = self._client.list_indexes()
            if self.config.collection_name in existing_indexes.names():
                self._index = self._client.Index(self.config.collection_name)
                logger.info(f"Connected to existing Pinecone index: {self.config.collection_name}")
            else:
                self._client.create_index(
                    name=self.config.collection_name,
                    dimension=self.config.dimension,
                    metric=VECTOR_STORE_METRIC,
                    spec=ServerlessSpec(cloud=VECTOR_STORE_CLOUD, region=VECTOR_STORE_REGION),
                )
                self._index = self._client.Index(self.config.collection_name)
                logger.info(f"Created new Pinecone index: {self.config.collection_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone index: {e}") from e

    def add_documents(self, documents: list[dict[str, Any]]) -> bool:
        try:
            texts = [doc.get("content", "") for doc in documents]
            embeddings = self.embedding_provider.encode(texts)

            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings, strict=False)):
                vector_id = f"doc_{i}_{int(time.time())}"
                vectors.append(
                    {
                        "id": vector_id,
                        "values": embedding.tolist(),
                        "metadata": {"content": doc.get("content", ""), **doc.get("metadata", {})},
                    }
                )

            batch_size = VECTOR_STORE_BATCH_SIZE
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self._index.upsert(vectors=batch)

            logger.info(f"Added {len(documents)} documents to Pinecone index")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {e}")
            return False

    def search(self, query_vector: np.ndarray, top_k: int = settings.vector_search_top_k) -> list[dict[str, Any]]:
        try:
            if query_vector.ndim > 1:
                query_vector = query_vector.flatten()

            results = self._index.query(
                vector=query_vector.tolist(), top_k=top_k, include_metadata=True
            )

            formatted_results = []
            for match in results.matches:
                formatted_results.append(
                    {
                        "content": match.metadata.get("content", ""),
                        "metadata": {k: v for k, v in match.metadata.items() if k != "content"},
                        "similarity": float(match.score),
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search Pinecone index: {e}")
            return []

    def search_text(self, query_text: str, top_k: int = settings.vector_search_top_k) -> list[dict[str, Any]]:
        try:
            query_embedding = self.embedding_provider.encode([query_text])[0]
            return self.search(query_embedding, top_k)
        except Exception as e:
            logger.error(f"Failed to search with text query: {e}")
            return []

    def get_stats(self) -> dict[str, Any]:
        try:
            stats = self._index.describe_index_stats()
            return {
                "document_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "store_type": "pinecone",
                "index_name": self.config.collection_name,
                "namespaces": list(stats.namespaces.keys()) if stats.namespaces else [],
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return {"error": str(e)}

    def delete_collection(self) -> bool:
        try:
            self._client.delete_index(self.config.collection_name)
            logger.info(f"Deleted Pinecone index: {self.config.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Pinecone index: {e}")
            return False


def create_vector_store(
    config: VectorStoreConfig, embedding_provider: EmbeddingProvider
) -> VectorStore:
    return PineconeVectorStore(config, embedding_provider)

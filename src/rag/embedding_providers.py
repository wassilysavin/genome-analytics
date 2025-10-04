import logging
from typing import Protocol

import numpy as np

from ..settings import settings
from ..constants.constants import *

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def encode(self, texts: list[str]) -> list[np.ndarray]: ...


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.dimension = EMBEDDING_DIMENSION_DEFAULT
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading Sentence Transformers model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Successfully loaded model: {self.model_name}")
            except ImportError as e:
                logger.error("sentence-transformers package not installed")
                logger.error(
                    "Failed to add documents to Pinecone: sentence-transformers package required. Install with: pip install sentence-transformers"
                )
                raise ImportError(
                    "sentence-transformers package required. "
                    "Install with: pip install sentence-transformers"
                ) from e
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise RuntimeError(f"Failed to load Sentence Transformers model: {e}") from e

        return self._model

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)

        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            embeddings = [emb.astype(np.float32) for emb in embeddings]
        else:
            embeddings = [np.array(emb, dtype=np.float32) for emb in embeddings]

        logger.info(f"Generated {len(embeddings)} embeddings using {self.model_name}")
        return embeddings


def create_embedding_provider(model_name: str = None) -> EmbeddingProvider:
    model_name = model_name or settings.embedding_model
    logger.info(f"Using {model_name} sentence transformer model")
    return SentenceTransformerEmbeddingProvider(model_name)

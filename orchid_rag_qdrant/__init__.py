"""Orchid Qdrant RAG plugin.

Registers ``qdrant`` vector and doc-store backends via entry points.
"""

from __future__ import annotations

import logging

__version__ = "0.0.0"
logger = logging.getLogger(__name__)


def _build_qdrant_reader(
    *,
    qdrant_url: str = "http://qdrant:6333",
    embedding_model: str = "text-embedding-3-small",
    **_settings: object,
) -> object:
    from orchid_ai.rag.embeddings import build_embeddings, get_embedding_dimension

    from .repository import QdrantRepository

    embeddings = build_embeddings(embedding_model)
    dimension = get_embedding_dimension(embedding_model)
    return QdrantRepository(
        url=qdrant_url,
        embeddings=embeddings,
        embedding_dimension=dimension,
    )


def _build_qdrant_doc_store(
    *,
    qdrant_url: str = "http://qdrant:6333",
    doc_store_collection: str = "__doc_store__",
    **_settings: object,
) -> object:
    from .doc_store import QdrantDocStore

    return QdrantDocStore(url=qdrant_url, collection_name=doc_store_collection)


def _register() -> None:
    try:
        from orchid_ai.rag.factory import (
            register_doc_store_backend,
            register_vector_backend,
        )

        register_vector_backend("qdrant", _build_qdrant_reader)
        register_doc_store_backend("qdrant", _build_qdrant_doc_store)
        logger.debug("[orchid-rag-qdrant] Registered backends")
    except ImportError:
        logger.debug("[orchid-rag-qdrant] Skipping registration (not in this orchid-ai version)")

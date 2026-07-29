"""Qdrant vector-store backend for Orchid.

Provides :class:`QdrantRepository` (read + write + admin) backed by
``qdrant_client.AsyncQdrantClient``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from orchid_ai.core.repository import (
    OrchidSearchResult,
    OrchidVectorStoreRepository,
)
from orchid_ai.core.scopes import SHARED_TENANT, OrchidRAGScope
from orchid_ai.core.sparse import OrchidSparseEncoder, OrchidSparseVector
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
)

logger = logging.getLogger(__name__)

# Stable v5 UUID namespace for point IDs — shared with ChromaRepository.
_POINT_ID_NAMESPACE = uuid.UUID("1f23d4a0-2b6e-44b9-9c5c-c2b7e3c8d1e0")


def build_qdrant_filter(scope: OrchidRAGScope, default_tenant: str = "default") -> Filter:
    """Build a Qdrant ``Filter`` with ``should`` clauses across all visible scope levels.

    Mirrors the hierarchical visibility model in :mod:`orchid_ai.rag.scopes`.
    """
    from qdrant_client.models import FieldCondition as _FC
    from qdrant_client.models import MatchValue as _MV

    tenant_id = scope.tenant_id or default_tenant
    clauses: list[Filter] = []

    # 1. Root common — tenant_id = "__shared__"
    clauses.append(Filter(must=[_FC(key="tenant_id", match=_MV(value=SHARED_TENANT))]))

    # 2. Tenant-level — tenant_id = T AND scope = "tenant"
    clauses.append(
        Filter(
            must=[
                _FC(key="tenant_id", match=_MV(value=tenant_id)),
                _FC(key="scope", match=_MV(value="tenant")),
            ]
        )
    )

    # 3. User-common — requires user_id
    if scope.user_id:
        clauses.append(
            Filter(
                must=[
                    _FC(key="tenant_id", match=_MV(value=tenant_id)),
                    _FC(key="user_id", match=_MV(value=scope.user_id)),
                    _FC(key="scope", match=_MV(value="user")),
                ]
            )
        )

    # 4. Chat-shared — requires user_id + chat_id
    if scope.user_id and scope.chat_id:
        clauses.append(
            Filter(
                must=[
                    _FC(key="tenant_id", match=_MV(value=tenant_id)),
                    _FC(key="user_id", match=_MV(value=scope.user_id)),
                    _FC(key="chat_id", match=_MV(value=scope.chat_id)),
                    _FC(key="scope", match=_MV(value="chat_shared")),
                ]
            )
        )

    # 5. Agent-private — requires user_id + chat_id + agent_id
    if scope.user_id and scope.chat_id and scope.agent_id:
        clauses.append(
            Filter(
                must=[
                    _FC(key="tenant_id", match=_MV(value=tenant_id)),
                    _FC(key="user_id", match=_MV(value=scope.user_id)),
                    _FC(key="chat_id", match=_MV(value=scope.chat_id)),
                    _FC(key="agent_id", match=_MV(value=scope.agent_id)),
                    _FC(key="scope", match=_MV(value="chat_agent")),
                ]
            )
        )

    return Filter(should=clauses)


def build_metadata_filter_clauses(
    metadata_filters: dict[str, Any],
) -> tuple[list[FieldCondition], list[FieldCondition]]:
    """Translate the metadata-filter mini-language into Qdrant ``must`` / ``must_not`` lists.

    Returns ``(must, must_not)`` where each element is a
    :class:`qdrant_client.models.FieldCondition`.
    """
    must: list[FieldCondition] = []
    must_not: list[FieldCondition] = []

    for key, value in metadata_filters.items():
        if key.startswith("_"):
            continue

        if isinstance(value, list):
            must.append(FieldCondition(key=key, match=MatchAny(any=value)))
            continue

        if isinstance(value, dict):
            if "not" in value:
                must_not.append(FieldCondition(key=key, match=MatchValue(value=value["not"])))
                continue
            if "contains" in value:
                must.append(FieldCondition(key=key, match=MatchValue(value=value["contains"])))
                continue

            # Range operators
            range_kwargs: dict[str, Any] = {}
            for op in ("gte", "lte", "gt", "lt"):
                if op in value:
                    range_kwargs[op] = value[op]
            if range_kwargs:
                # ISO-8601 strings → DatetimeRange; everything else → Range
                first_val = next(iter(range_kwargs.values()))
                if isinstance(first_val, str):
                    must.append(FieldCondition(key=key, range=DatetimeRange(**range_kwargs)))
                else:
                    must.append(FieldCondition(key=key, range=Range(**range_kwargs)))
                continue

            raise ValueError(
                f"Unknown metadata filter operator(s) for {key!r}: {sorted(value)}. "
                f"Allowed: gte/lte/gt/lt/contains/not."
            )

        # Scalar exact-match
        must.append(FieldCondition(key=key, match=MatchValue(value=value)))

    return must, must_not


def infer_payload_index_types(metadata_filters: dict[str, Any]) -> dict[str, str]:
    """Infer Qdrant payload-index types from the values in a metadata-filter dict.

    Returns a mapping ``field_name → index_type`` where index_type is one of
    ``keyword``, ``integer``, ``float``, ``bool``, ``datetime``.
    """
    types: dict[str, str] = {}
    for key, value in metadata_filters.items():
        if key.startswith("_"):
            continue
        inferred = _infer_type(value)
        if inferred:
            types[key] = inferred
    return types


def _infer_type(value: Any) -> str | None:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "keyword"
    if isinstance(value, list):
        for item in value:
            if item is not None:
                return _infer_type(item)
        return None
    if isinstance(value, dict):
        if "not" in value:
            return _infer_type(value["not"])
        if "contains" in value:
            return _infer_type(value["contains"])
        # Range → infer from first bound value
        for op in ("gte", "lte", "gt", "lt"):
            if op in value:
                v = value[op]
                if isinstance(v, str):
                    return "datetime"
                return _infer_type(v)
        return None
    return None


class QdrantRepository(OrchidVectorStoreRepository):
    """Qdrant-backed vector store with per-tenant isolation.

    Each ``namespace`` maps to a Qdrant **collection**.
    Tenant isolation is enforced via a ``tenant_id`` payload field on every
    document, and all reads filter on the visible scope levels.
    """

    supports_scope_promotion = True

    def __init__(
        self,
        *,
        url: str,
        embeddings: Embeddings,
        embedding_dimension: int = 1536,
        default_tenant: str = "default",
        sparse_encoder: OrchidSparseEncoder | None = None,
        client: Any | None = None,
    ):
        from qdrant_client import AsyncQdrantClient

        self._client = client or AsyncQdrantClient(url=url)
        self._embeddings = embeddings
        self._embedding_dimension = embedding_dimension
        self._default_tenant = default_tenant
        self._sparse_encoder = sparse_encoder
        # Tracks which collections have been verified / created.
        self._verified_collections: set[str] = set()
        # Tracks ensured payload indexes per (namespace, field_name).
        self._ensured_indexes: set[tuple[str, str]] = set()

    # ── OrchidVectorReader ──────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        namespace: str,
        k: int = 5,
        scope: OrchidRAGScope | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[OrchidSearchResult]:
        """Retrieve the *k* most relevant documents for *query* in *namespace*."""
        await self._ensure_collection(namespace)

        query_embedding = await self._embeddings.aembed_query(query)

        scope_filter = build_qdrant_filter(scope, self._default_tenant) if scope else None
        meta_must, meta_must_not = build_metadata_filter_clauses(metadata_filters) if metadata_filters else ([], [])

        if meta_must or meta_must_not:
            await self._ensure_inferred_payload_indexes(namespace, metadata_filters)

        if meta_must or meta_must_not:
            # When metadata filters are present, ``should`` alone becomes
            # scoring-only.  Wrap the scope ``should`` clauses inside a
            # ``must`` group so they stay hard OR filters, ANDed with the
            # metadata conditions.
            qdrant_filter = Filter(
                must=[scope_filter, *meta_must] if scope_filter else meta_must,
                must_not=meta_must_not,
            )
        else:
            qdrant_filter = scope_filter  # scope ``should`` acts as hard OR filter on its own

        results = await self._client.query_points(
            collection_name=namespace,
            query=query_embedding,
            limit=k,
            query_filter=qdrant_filter if qdrant_filter else None,
            with_payload=True,
        )

        out: list[OrchidSearchResult] = []
        for point in results.points:
            meta = dict(point.payload or {})
            out.append(
                OrchidSearchResult(
                    document=Document(
                        id=str(meta.get("doc_id", point.id)),
                        page_content=str(meta.get("content", "")),
                        metadata=meta,
                    ),
                    score=getattr(point, "score", 0.0),
                )
            )
        return out

    async def retrieve_sparse(
        self,
        query_sparse: OrchidSparseVector,
        namespace: str,
        k: int = 5,
        scope: OrchidRAGScope | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[OrchidSearchResult]:
        """Retrieve via the sparse vector lane (requires a named sparse vector)."""
        if self._sparse_encoder is None:
            raise NotImplementedError("QdrantRepository sparse retrieval requires a sparse_encoder.")
        await self._ensure_collection(namespace)

        scope_filter = build_qdrant_filter(scope, self._default_tenant) if scope else None
        meta_must, meta_must_not = build_metadata_filter_clauses(metadata_filters) if metadata_filters else ([], [])

        if meta_must or meta_must_not:
            qdrant_filter = Filter(
                must=[scope_filter, *meta_must] if scope_filter else meta_must,
                must_not=meta_must_not,
            )
        else:
            qdrant_filter = scope_filter

        results = await self._client.query_points(
            collection_name=namespace,
            query=query_sparse,
            using="sparse",
            limit=k,
            query_filter=qdrant_filter if qdrant_filter else None,
            with_payload=True,
        )

        out: list[OrchidSearchResult] = []
        for point in results.points:
            meta = dict(point.payload or {})
            out.append(
                OrchidSearchResult(
                    document=Document(
                        id=str(meta.get("doc_id", point.id)),
                        page_content=str(meta.get("content", "")),
                        metadata=meta,
                    ),
                    score=getattr(point, "score", 0.0),
                )
            )
        return out

    async def lookup_cached_tool_results(
        self,
        namespace: str,
        scope: OrchidRAGScope,
        tool_name: str,
        min_injected_at: float,
    ) -> str | None:
        """Lookup cached tool results by metadata."""
        await self._ensure_collection(namespace)

        q_filter = build_qdrant_filter(scope, self._default_tenant)
        meta_must, _ = build_metadata_filter_clauses({"tool_name": tool_name, "injected_at": {"gte": min_injected_at}})
        if meta_must:
            # Wrap scope inside must alongside metadata so that the scope's
            # should clauses still act as a hard OR filter (nested Filter
            # inside must preserves OR semantics for the group).
            q_filter = Filter(must=[q_filter, *meta_must])

        results = await self._client.query_points(
            collection_name=namespace,
            query=None,
            limit=1,
            query_filter=q_filter,
            with_payload=True,
        )
        if results.points:
            payload = results.points[0].payload or {}
            return str(payload.get("content", ""))
        return None

    # ── OrchidVectorWriter ──────────────────────────────────────────

    async def index(
        self,
        documents: list[Document],
        namespace: str,
    ) -> None:
        """Index documents — creates the collection if it doesn't exist."""
        await self._ensure_collection(namespace)
        if not documents:
            return

        embeddings = await self._embeddings.aembed_documents([doc.page_content for doc in documents])
        points = self._build_points(documents, embeddings)
        await self._client.upsert(collection_name=namespace, points=points)
        logger.info("[Qdrant] indexed %d documents in '%s'", len(documents), namespace)

    async def upsert(
        self,
        documents: list[Document],
        namespace: str,
    ) -> None:
        """Insert or update documents (idempotent)."""
        await self._ensure_collection(namespace)
        if not documents:
            return

        embeddings = await self._embeddings.aembed_documents([doc.page_content for doc in documents])
        points = self._build_points(documents, embeddings)
        await self._client.upsert(collection_name=namespace, points=points)
        logger.info("[Qdrant] upserted %d documents in '%s'", len(documents), namespace)

    async def delete(
        self,
        document_ids: list[str],
        namespace: str,
    ) -> None:
        """Remove documents by ID from the namespace."""
        if not document_ids:
            return
        await self._ensure_collection(namespace)
        point_ids = [str(uuid.uuid5(_POINT_ID_NAMESPACE, doc_id)) for doc_id in document_ids]
        await self._client.delete(collection_name=namespace, points_selector=point_ids)
        logger.info("[Qdrant] deleted %d documents from '%s'", len(document_ids), namespace)

    # ── OrchidVectorStoreAdmin ──────────────────────────────────────

    async def ensure_collections(self, namespaces: list[str]) -> None:
        """Pre-create collections at startup (called from lifespan)."""
        for ns in namespaces:
            await self._ensure_collection(ns)

    async def _ensure_collection(self, namespace: str) -> None:
        """Create the collection if missing."""
        if namespace in self._verified_collections:
            return
        from qdrant_client.models import Distance, VectorParams

        exists = await self._client.collection_exists(namespace)
        if not exists:
            await self._client.create_collection(
                collection_name=namespace,
                vectors_config=VectorParams(
                    size=self._embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "[Qdrant] created collection '%s' (dim=%d)",
                namespace,
                self._embedding_dimension,
            )
        self._verified_collections.add(namespace)

    async def ensure_payload_indexes(
        self,
        namespace: str,
        payload_indexes: dict[str, str],
    ) -> None:
        """Create payload indexes for metadata-filter fields.

        ``payload_indexes`` is a mapping of ``field_name → index_type``
        where index_type is one of ``keyword``, ``integer``, ``float``,
        ``bool``, ``datetime``.

        Calls are idempotent — already-created indexes are cached in
        ``_ensured_indexes`` and skipped on repeat invocations.
        Qdrant errors (e.g. schema clashes) are logged as warnings
        rather than raised.
        """
        if not payload_indexes:
            return
        await self._ensure_collection(namespace)
        for field_name, index_type in payload_indexes.items():
            key = (namespace, field_name)
            if key in self._ensured_indexes:
                continue
            try:
                await self._client.create_payload_index(
                    collection_name=namespace,
                    field_name=field_name,
                    field_schema=index_type,
                )
                self._ensured_indexes.add(key)
                logger.debug(
                    "[Qdrant] payload index created: %s.%s (%s)",
                    namespace,
                    field_name,
                    index_type,
                )
            except Exception as exc:
                logger.warning(
                    "[Qdrant] payload index creation failed for %s.%s: %s",
                    namespace,
                    field_name,
                    exc,
                )

    async def _ensure_inferred_payload_indexes(
        self,
        namespace: str,
        metadata_filters: dict[str, Any],
    ) -> None:
        """Infer and create payload indexes from metadata filter operands."""
        indexes = infer_payload_index_types(metadata_filters)
        await self.ensure_payload_indexes(namespace, indexes)

    # ── Scope promotion ─────────────────────────────────────────────

    async def promote_scope(
        self,
        *,
        namespace: str,
        source_filter: Any,
        new_scope_fields: dict,
    ) -> int:
        """Promote data to a broader scope (e.g. chat → user for sharing).

        Scrolls matching points, clones them with ``new_scope_fields``,
        and upserts the clones.  Returns the number of points promoted.
        """
        await self._ensure_collection(namespace)
        points: list[PointStruct] = []
        offset: str | None = None
        while True:
            batch, offset = await self._client.scroll(
                collection_name=namespace,
                filter=source_filter,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in batch:
                payload = dict(point.payload or {})
                payload.update(new_scope_fields)
                points.append(
                    PointStruct(
                        id=point.id,
                        payload=payload,
                        vector=point.vector,
                    )
                )
            if offset is None:
                break

        if points:
            await self._client.upsert(collection_name=namespace, points=points)
            logger.info(
                "[Qdrant] promoted %d points in '%s'",
                len(points),
                namespace,
            )
        return len(points)

    # ── Helpers ─────────────────────────────────────────────────────

    def _build_points(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[PointStruct]:
        points: list[PointStruct] = []
        for i, doc in enumerate(documents):
            if not doc.id:
                logger.debug("[Qdrant] document missing id — falling back to page_content hash")
            point_id = str(uuid.uuid5(_POINT_ID_NAMESPACE, doc.id or doc.page_content))
            payload = dict(doc.metadata)
            payload["doc_id"] = doc.id
            payload["content"] = doc.page_content
            payload.setdefault("tenant_id", self._default_tenant)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[i],
                    payload=payload,
                )
            )
        return points

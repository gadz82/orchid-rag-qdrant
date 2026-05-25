"""Qdrant-backed parent-document store for Orchid.

Uses a dedicated Qdrant collection to store parent documents by stable ID.
Each document's point ID is a deterministic UUID derived from ``doc_id``
so re-putting the same ID is idempotent.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client.models import PointStruct

from orchid_ai.core.doc_store import OrchidDocStore

logger = logging.getLogger(__name__)

# Stable v5 UUID namespace for point IDs — shared with QdrantRepository.
_POINT_ID_NAMESPACE = uuid.UUID("1f23d4a0-2b6e-44b9-9c5c-c2b7e3c8d1e0")


class QdrantDocStore(OrchidDocStore):
    """Qdrant-backed parent-document store.

    Uses a dedicated collection to store parent documents by stable ID.
    Each document's point ID is a deterministic UUID derived from
    ``doc_id`` so re-putting the same ID is idempotent.
    """

    def __init__(
        self,
        *,
        url: str,
        collection_name: str = "__doc_store__",
        client: Any | None = None,
    ):
        from qdrant_client import AsyncQdrantClient

        self._client = client or AsyncQdrantClient(url=url)
        self._collection_name = collection_name

    async def put(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        await self._ensure_collection()
        point_id = _doc_id_to_uuid(doc_id)
        payload = dict(metadata)
        payload["doc_id"] = doc_id
        payload["content"] = content
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    payload=payload,
                    vector={},
                )
            ],
        )

    async def get(self, doc_id: str) -> tuple[str, dict[str, Any]] | None:
        await self._ensure_collection()
        point_id = _doc_id_to_uuid(doc_id)
        results = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=[point_id],
            with_payload=True,
        )
        if not results:
            return None
        payload = dict(results[0].payload or {})
        content = payload.pop("content", "")
        payload.pop("doc_id", None)
        return str(content), payload

    async def get_many(
        self, doc_ids: list[str]
    ) -> dict[str, tuple[str, dict[str, Any]]]:
        if not doc_ids:
            return {}
        await self._ensure_collection()
        point_ids = [_doc_id_to_uuid(d) for d in doc_ids]
        results = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=point_ids,
            with_payload=True,
        )
        out: dict[str, tuple[str, dict[str, Any]]] = {}
        for res in results:
            payload = dict(res.payload or {})
            d_id = payload.get("doc_id")
            if not d_id:
                continue
            content = payload.pop("content", "")
            payload.pop("doc_id", None)
            out[str(d_id)] = (str(content), payload)
        return out

    async def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        exists = await self._client.collection_exists(self._collection_name)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )


def _doc_id_to_uuid(doc_id: str) -> str:
    """Deterministic UUID from a doc_id."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, doc_id))

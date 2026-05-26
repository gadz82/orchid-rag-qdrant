"""Tests for QdrantRepository.promote_scope."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import PointStruct

from orchid_ai.core.scopes import OrchidRAGScope
from orchid_rag_qdrant.repository import QdrantRepository


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()
    client.scroll = AsyncMock()
    client.upsert = AsyncMock()
    return client


def _repo(client) -> QdrantRepository:
    embeddings = MagicMock()
    embeddings.aembed_query = AsyncMock(return_value=[0.0])
    return QdrantRepository(
        url="http://x",
        embeddings=embeddings,
        embedding_dimension=1,
        client=client,
    )


def _point(id_num: int, *, payload: dict | None = None, vector: list[float] | None = None) -> MagicMock:
    p = MagicMock()
    p.id = id_num
    p.payload = payload
    p.vector = vector or [0.0]
    return p


class TestPromoteScope:
    @pytest.mark.asyncio
    async def test_promotes_points_with_updated_payload(self):
        client = _mock_client()
        repo = _repo(client)

        client.scroll.side_effect = [
            ([_point(1, payload={"scope": "chat_shared"}, vector=[1.0, 2.0])], "offset1"),
            ([_point(2, payload={"scope": "chat_shared"}, vector=[3.0, 4.0])], None),
        ]

        source_filter = MagicMock()
        new_fields = {"scope": "user", "user_id": "u-1"}

        result = await repo.promote_scope(
            namespace="ns",
            source_filter=source_filter,
            new_scope_fields=new_fields,
        )

        assert result == 2
        assert client.upsert.await_count == 1
        call_args = client.upsert.await_args
        assert call_args.kwargs["collection_name"] == "ns"

        points: list[PointStruct] = call_args.kwargs["points"]
        assert len(points) == 2

        assert points[0].id == 1
        assert points[0].payload == {"scope": "user", "user_id": "u-1"}
        assert points[0].vector == [1.0, 2.0]

        assert points[1].id == 2
        assert points[1].payload == {"scope": "user", "user_id": "u-1"}
        assert points[1].vector == [3.0, 4.0]

    @pytest.mark.asyncio
    async def test_promotes_no_points_when_scroll_empty(self):
        client = _mock_client()
        repo = _repo(client)

        client.scroll.side_effect = [([], None)]

        result = await repo.promote_scope(
            namespace="ns",
            source_filter=MagicMock(),
            new_scope_fields={"scope": "tenant"},
        )

        assert result == 0
        client.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_promote_scope_scrolls_with_vectors(self):
        client = _mock_client()
        repo = _repo(client)

        client.scroll.side_effect = [([], None)]

        await repo.promote_scope(
            namespace="ns",
            source_filter=MagicMock(),
            new_scope_fields={"scope": "tenant"},
        )

        client.scroll.assert_awaited_once()
        scroll_kwargs = client.scroll.await_args.kwargs
        assert scroll_kwargs["with_vectors"] is True

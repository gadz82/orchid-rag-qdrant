# orchid-rag-qdrant — AI Context

## What This Package Is

`orchid-rag-qdrant` is the Qdrant vector-backend plugin for the Orchid AI
framework. It provides:

- `QdrantRepository` — implements `OrchidVectorStoreRepository` (read + write + admin)
- `QdrantDocStore` — implements `OrchidDocStore` (parent-document storage)
- `build_qdrant_filter` — translates `OrchidRAGScope` into Qdrant-native `Filter`
- `build_metadata_filter_clauses` + `infer_payload_index_types` — metadata-filter mini-language translation

## Auto-Registration

The package registers itself via Python `importlib.metadata` entry points:

```toml
[project.entry-points."orchid.vector_backends"]
qdrant = "orchid_rag_qdrant:_register"

[project.entry-points."orchid.doc_store_backends"]
qdrant = "orchid_rag_qdrant:_register"
```

No manual `register_vector_backend()` calls are needed by integrators.

## Key Files

| File | Purpose |
|------|---------|
| `repository.py` | `QdrantRepository`, filter builders, index inference |
| `doc_store.py` | `QdrantDocStore`, `_doc_id_to_uuid` |
| `__init__.py` | Entry-point `_register()` callable |

## Testing

Tests require `qdrant-client` but do **not** require a live Qdrant server —
all unit tests mock `AsyncQdrantClient`.

```bash
cd orchid-rag-qdrant
pip install -e ".[dev]"
pytest tests/ -x
```

## Common Pitfalls

- Embedding dimension mismatch (768 vs 1536 vs 3072) causes silent retrieval failures. Switching models requires re-creating Qdrant collections.
- The `_POINT_ID_NAMESPACE` UUID must stay stable — it deterministically maps `doc_id` → Qdrant point ID across sessions.

# orchid-rag-qdrant

Qdrant vector and doc-store backend for the [Orchid](https://github.com/gadz82/orchid) multi-agent AI framework.

## Installation

```bash
pip install orchid-rag-qdrant
```

## Usage

No code changes are required. Once installed, the package auto-registers
``qdrant`` in Orchid's vector and doc-store backend registries via Python
entry points.

```yaml
# orchid.yml
vector_backend: qdrant
qdrant_url: http://qdrant:6333
```

## Requirements

- Python 3.11+
- A running Qdrant instance (or use the embedded Qdrant client for testing)

## Documentation

See the main [Orchid framework docs](https://github.com/gadz82/orchid) for
full configuration reference and examples.

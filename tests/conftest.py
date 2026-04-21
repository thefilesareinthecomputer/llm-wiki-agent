"""
Pytest configuration and fixtures.
"""

import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient


class FakeEmbeddingFunction:
    """Deterministic fake embeddings for tests. 768-dim vectors derived from
    a hash of the input text -- similar inputs get similar vectors.
    Zero Gemini API calls.
    """

    def name(self):
        return "fake_768d"

    def __call__(self, input):
        out = []
        for text in input:
            h = hashlib.sha256(text.encode()).digest()
            seed = int.from_bytes(h[:4], "big")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(768).astype(np.float32)
            v /= np.linalg.norm(v) or 1.0
            out.append(v.tolist())
        return out


@pytest.fixture(scope="function")
def client_with_init(tmp_path):
    """Create test client with fake embeddings (no Gemini API calls).

    Uses a TEMPORARY LanceDB directory so tests never overwrite
    the production embeddings.
    """
    import lancedb
    from web.app import app, set_components
    from knowledge.index import KBIndex
    from models.gateway import ModelGateway
    from memory.store import ConversationStore
    from agent.tools import KBTools

    # Create components manually with fake embeddings
    gateway = ModelGateway()
    kb_index = KBIndex()

    # Use temp LanceDB dir — never touch production embeddings
    kb_index.db = lancedb.connect(str(tmp_path / "lancedb"))
    kb_index._embedding_fn = FakeEmbeddingFunction()

    store = ConversationStore()
    store.initialize()
    tools = KBTools(kb_index=kb_index)

    # Set components so lifespan skips initialize_app
    set_components(gateway, kb_index, store, tools)

    # Build index with fake embeddings (fast, no API calls)
    kb_index.build_index()

    with TestClient(app, raise_server_exceptions=True) as test_client:
        yield test_client


@pytest.fixture
def temp_kb_dir(tmp_path):
    """Create temporary knowledge base directories."""
    kb_dir = tmp_path / "knowledge"
    canon_dir = tmp_path / "canon"
    kb_dir.mkdir()
    canon_dir.mkdir()
    return kb_dir, canon_dir


@pytest.fixture
def mock_model_gateway():
    """Mock ModelGateway for testing."""
    from unittest.mock import AsyncMock, MagicMock

    gateway = MagicMock()
    gateway.chat = AsyncMock(return_value="Mock response")
    gateway.test_connection = AsyncMock(return_value=True)
    return gateway


@pytest.fixture
def sample_md_content():
    """Sample markdown content with frontmatter."""
    return """---
created: 2026-04-13T18:00:00Z
created_by: user
updated: 2026-04-13T19:00:00Z
last_modified_by: llm_wiki_agent
tags: [test, example]
---

# Test Document

This is test content with `inline code` and a code block:

```python
def hello():
    print("Hello, World!")
```
"""
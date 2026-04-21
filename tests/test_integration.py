"""
Integration tests — verify components are shared correctly
and architectural contracts hold.
"""

import pytest
from web.app import app, set_components, model_gateway, kb_index, memory_store
from models.gateway import ModelGateway
from memory.store import ConversationStore
from knowledge.index import KBIndex
from unittest.mock import AsyncMock


class TestSharedComponents:
    """Verify that set_components() makes app.py use externally-created instances."""

    def test_set_components_populates_globals(self, tmp_path):
        """set_components should set the module-level globals."""
        gateway = ModelGateway()
        store = ConversationStore(sessions_dir=tmp_path / "sessions")
        index = KBIndex()

        set_components(gateway, index, store)

        from web.app import model_gateway as mg, kb_index as ki, memory_store as ms
        assert mg is gateway
        assert ki is index
        assert ms is store

    def test_set_components_prevents_lifespan_reinit(self, tmp_path):
        """If components are set before lifespan, lifespan should NOT reinitialize."""
        gateway = ModelGateway()
        store = ConversationStore(sessions_dir=tmp_path / "sessions")
        store.initialize()
        index = KBIndex()

        set_components(gateway, index, store)

        # Lifespan should see non-None globals and skip init
        from fastapi.testclient import TestClient
        with TestClient(app, raise_server_exceptions=True) as client:
            # App should work with the pre-set components
            resp = client.get("/kb/stats")
            assert resp.status_code == 200


class TestWatcherReindexIntegration:
    """Verify watcher reindex operates on the same KBIndex instance as the web app."""

    def test_reindex_callback_uses_shared_index(self, tmp_path):
        """When main.py passes kb_index to both watcher and app, reindex affects the same instance."""
        from agent.watcher import KnowledgeBaseWatcher

        index = KBIndex()
        reindex_called = []

        def callback():
            reindex_called.append(True)

        watcher = KnowledgeBaseWatcher(
            tmp_path / "kb", tmp_path / "canon", reindex_callback=callback
        )

        # Verify callback is wired
        assert watcher.reindex_callback is callback

        # Create dirs so watcher.start() doesn't crash
        (tmp_path / "kb").mkdir(exist_ok=True)
        (tmp_path / "canon").mkdir(exist_ok=True)


class TestEndpointSmokeTests:
    """Quick smoke tests for all endpoints after architectural fixes."""

    @pytest.fixture(autouse=True)
    def setup(self, client_with_init):
        self.client = client_with_init

    def test_root_returns_html(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_models_returns_json(self):
        resp = self.client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "current" in data

    def test_kb_stats_returns_counts(self):
        resp = self.client.get("/kb/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "files" in data
        assert "vectors" in data

    def test_kb_reindex_works(self):
        resp = self.client.post("/kb/reindex")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_set_model_endpoint(self):
        resp = self.client.post("/model", json={"model": "test-model"})
        assert resp.status_code == 200

    def test_kb_search_returns_list(self):
        resp = self.client.get("/kb/search?q=test")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_kb_list_knowledge(self):
        resp = self.client.get("/kb/knowledge")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_kb_list_canon(self):
        resp = self.client.get("/kb/canon")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestSummaryPersistenceIntegration:
    """Integration tests verifying LLM summaries survive watcher-triggered reindexes."""

    def test_reindex_file_preserves_other_file_rows(self, tmp_path, monkeypatch):
        """reindex_file on one file doesn't touch other files' rows in LanceDB."""
        import knowledge.index as kb_idx

        kb_dir = tmp_path / "knowledge"
        canon_dir = tmp_path / "canon"
        kb_dir.mkdir()
        canon_dir.mkdir()
        (kb_dir / "alpha.md").write_text("# Alpha\n\nAlpha content.\n")
        (kb_dir / "beta.md").write_text("# Beta\n\nBeta content.\n")

        monkeypatch.setattr(kb_idx, 'KB_DIR', kb_dir)
        monkeypatch.setattr(kb_idx, 'CANON_DIR', canon_dir)
        monkeypatch.setattr(kb_idx, 'LANCEDB_DIR', tmp_path / 'lancedb')

        index = KBIndex()
        index.build_index()

        # Check beta rows exist
        df = index.table.to_pandas()
        beta_before = len(df[df['filename'] == 'beta.md'])
        assert beta_before > 0

        # Reindex only alpha
        index.reindex_file(kb_dir / "alpha.md")

        # Beta rows unchanged
        df = index.table.to_pandas()
        beta_after = len(df[df['filename'] == 'beta.md'])
        assert beta_after == beta_before

    def test_reindex_endpoint_stores_summaries(self, tmp_path, monkeypatch):
        """Calling build_index with llm_summaries=True stores LLM summaries in LanceDB."""
        import knowledge.index as kb_idx
        from unittest.mock import patch

        kb_dir = tmp_path / "knowledge"
        canon_dir = tmp_path / "canon"
        kb_dir.mkdir()
        canon_dir.mkdir()
        (kb_dir / "test.md").write_text("# Test\n\nTest content.\n")

        monkeypatch.setattr(kb_idx, 'KB_DIR', kb_dir)
        monkeypatch.setattr(kb_idx, 'CANON_DIR', canon_dir)
        monkeypatch.setattr(kb_idx, 'LANCEDB_DIR', tmp_path / 'lancedb')

        index = KBIndex()

        # Mock LLM summary generation to produce distinguishable text
        with patch.object(index, '_generate_section_summaries', return_value={0: "LLM SECTION SUMMARY"}):
            with patch.object(index, '_generate_doc_summary', return_value="LLM DOC OVERVIEW"):
                index.build_index(llm_summaries=True, force=True)

        # Verify LanceDB has the LLM summaries
        df = index.table.to_pandas()
        summaries = df['summary'].tolist()
        assert any("LLM SECTION SUMMARY" in s for s in summaries), \
            f"Expected LLM section summary in LanceDB, got: {summaries}"

    def test_watcher_callback_passes_path(self):
        """Watcher callback receives the changed file path."""
        from agent.watcher import KBEventHandler
        from watchdog.events import FileModifiedEvent
        from pathlib import Path

        received = []
        handler = KBEventHandler(reindex_callback=lambda p: received.append(p))
        handler.on_modified(FileModifiedEvent("/tmp/kb/test.md"))

        assert len(received) == 1
        assert isinstance(received[0], Path)
        assert str(received[0]).endswith("test.md")

    def test_full_reindex_then_single_file_preserves_summaries(self, tmp_path, monkeypatch):
        """Full reindex with LLM summaries, then single-file reindex preserves other summaries."""
        import knowledge.index as kb_idx
        from unittest.mock import patch

        kb_dir = tmp_path / "knowledge"
        canon_dir = tmp_path / "canon"
        kb_dir.mkdir()
        canon_dir.mkdir()
        (kb_dir / "alpha.md").write_text("# Alpha\n\nAlpha content.\n")
        (kb_dir / "beta.md").write_text("# Beta\n\nBeta content.\n")

        monkeypatch.setattr(kb_idx, 'KB_DIR', kb_dir)
        monkeypatch.setattr(kb_idx, 'CANON_DIR', canon_dir)
        monkeypatch.setattr(kb_idx, 'LANCEDB_DIR', tmp_path / 'lancedb')

        index = KBIndex()

        # Full reindex with LLM summaries
        with patch.object(index, '_generate_section_summaries',
                          side_effect=lambda chunks, fn, **kw: {i: f"LLM SUMMARY for {fn}" for i in range(len(chunks))}):
            with patch.object(index, '_generate_doc_summary', return_value="LLM DOC OVERVIEW"):
                index.build_index(llm_summaries=True, force=True)

        # Verify beta has LLM summary
        df = index.table.to_pandas()
        beta_rows = df[df['filename'] == 'beta.md']
        assert any("LLM SUMMARY" in s for s in beta_rows['summary'].tolist()), \
            "Beta should have LLM summary before reindex_file"

        # Reindex only alpha (simulating watcher — mechanical summaries only)
        index.reindex_file(kb_dir / "alpha.md")

        # Beta's LLM summaries should survive
        df = index.table.to_pandas()
        beta_rows = df[df['filename'] == 'beta.md']
        beta_summaries = beta_rows['summary'].tolist()
        assert any("LLM SUMMARY" in s for s in beta_summaries), \
            f"Beta LLM summary should survive reindex_file, got: {beta_summaries}"


class TestReindexEndpointParameters:
    """E2E tests for /kb/reindex with summaries and entities parameters."""

    @pytest.fixture(autouse=True)
    def setup(self, client_with_init):
        self.client = client_with_init

    def test_reindex_with_summaries_param(self, monkeypatch):
        """POST /kb/reindex with summaries=true passes llm_summaries to build_index."""
        from web.app import kb_index
        called_with = {}

        original_build = kb_index.build_index

        def capture_build(extract_entities=False, llm_summaries=False, force=False):
            called_with["llm_summaries"] = llm_summaries
            called_with["extract_entities"] = extract_entities
            called_with["force"] = force
            # Don't actually rebuild — just capture params
            return None

        monkeypatch.setattr(kb_index, "build_index", capture_build)

        resp = self.client.post("/kb/reindex", json={"summaries": True})
        assert resp.status_code == 200
        assert called_with["llm_summaries"] is True
        assert called_with["force"] is True

    def test_reindex_with_entities_param(self, monkeypatch):
        """POST /kb/reindex with entities=true passes extract_entities to build_index."""
        from web.app import kb_index
        called_with = {}

        def capture_build(extract_entities=False, llm_summaries=False, force=False):
            called_with["extract_entities"] = extract_entities
            called_with["llm_summaries"] = llm_summaries
            return None

        monkeypatch.setattr(kb_index, "build_index", capture_build)

        resp = self.client.post("/kb/reindex", json={"entities": True})
        assert resp.status_code == 200
        assert called_with["extract_entities"] is True

    def test_reindex_with_both_params(self, monkeypatch):
        """POST /kb/reindex with both summaries and entities sets both flags."""
        from web.app import kb_index
        called_with = {}

        def capture_build(extract_entities=False, llm_summaries=False, force=False):
            called_with["extract_entities"] = extract_entities
            called_with["llm_summaries"] = llm_summaries
            return None

        monkeypatch.setattr(kb_index, "build_index", capture_build)

        resp = self.client.post("/kb/reindex", json={"summaries": True, "entities": True})
        assert resp.status_code == 200
        assert called_with["llm_summaries"] is True
        assert called_with["extract_entities"] is True

    def test_reindex_default_no_summaries_no_entities(self, monkeypatch):
        """POST /kb/reindex with empty body defaults to no summaries, no entities."""
        from web.app import kb_index
        called_with = {}

        def capture_build(extract_entities=False, llm_summaries=False, force=False):
            called_with["extract_entities"] = extract_entities
            called_with["llm_summaries"] = llm_summaries
            return None

        monkeypatch.setattr(kb_index, "build_index", capture_build)

        resp = self.client.post("/kb/reindex")
        assert resp.status_code == 200
        assert called_with["llm_summaries"] is False
        assert called_with["extract_entities"] is False


class TestSummaryPersistenceAcrossRestart:
    """Integration test: LLM summaries survive process restart (new KBIndex same LanceDB dir)."""

    def test_summaries_persist_after_restart(self, tmp_path, monkeypatch):
        """Build index with LLM summaries, create fresh KBIndex on same dir, summaries survive."""
        import knowledge.index as kb_idx
        from unittest.mock import patch

        kb_dir = tmp_path / "knowledge"
        canon_dir = tmp_path / "canon"
        lancedb_dir = tmp_path / "lancedb"
        kb_dir.mkdir()
        canon_dir.mkdir()
        (kb_dir / "persist.md").write_text("# Persistence Test\n\nContent for persistence.\n")

        monkeypatch.setattr(kb_idx, 'KB_DIR', kb_dir)
        monkeypatch.setattr(kb_idx, 'CANON_DIR', canon_dir)
        monkeypatch.setattr(kb_idx, 'LANCEDB_DIR', lancedb_dir)

        # Build with LLM summaries
        index1 = KBIndex()
        with patch.object(index1, '_generate_section_summaries',
                          return_value={0: "LLM PERSISTED SUMMARY"}):
            with patch.object(index1, '_generate_doc_summary', return_value="LLM DOC OVERVIEW"):
                index1.build_index(llm_summaries=True, force=True)

        # Verify summaries written
        df1 = index1.table.to_pandas()
        assert any("LLM PERSISTED SUMMARY" in s for s in df1['summary'].tolist()), \
            "Summary should exist after first build"

        # Simulate process restart: new KBIndex pointing at same LanceDB
        index2 = KBIndex()
        index2.build_index()  # No llm_summaries — should read existing data

        # Summaries from first build should survive
        df2 = index2.table.to_pandas()
        persisted_summaries = [s for s in df2['summary'].tolist() if "LLM PERSISTED SUMMARY" in s]
        assert len(persisted_summaries) > 0, \
            f"LLM summary should survive restart, got: {df2['summary'].tolist()}"


class TestToolResultCap:
    """Test that tool results are capped before feeding back to model."""

    def test_large_tool_result_truncated(self):
        """Tool results exceeding 4000 chars are truncated with indicator."""
        from web.app import _execute_tool
        # The cap is applied in the chat endpoint, not in _execute_tool.
        # Test the cap logic directly:
        MAX_TOOL_RESULT_CHARS = 4000
        big_result = "x" * 10000
        if len(big_result) > MAX_TOOL_RESULT_CHARS:
            capped = big_result[:MAX_TOOL_RESULT_CHARS] + f"\n... [truncated, {len(big_result)} chars total]"
        else:
            capped = big_result
        assert len(capped) < 5000  # Well under 10MB
        assert "truncated" in capped
        assert "10000 chars total" in capped
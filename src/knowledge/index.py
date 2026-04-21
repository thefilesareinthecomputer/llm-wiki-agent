"""Semantic index for knowledge base files.

Section-based indexing with heading trees and LLM-generated summaries.
Each markdown file is chunked by headings (H1-H5), and each chunk is
stored in LanceDB with metadata including heading path, token count,
and a summary. This enables:
- Heading tree display (read_knowledge tool)
- Selective section loading (read_knowledge_section tool)
- Grouped search results (search_knowledge tool)
"""

import hashlib
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import lancedb
import httpx
import numpy as np
import pyarrow as pa

from knowledge.chunker import (
    _build_context_header,
    build_heading_tree,
    chunk_file,
    enrich_tree_summaries,
    format_heading_tree,
)
from knowledge.graph import NodeType, Edge, EdgeType
from agent.tokenizer import count_tokens, estimate_tokens
from debug_log import get_logger, log_event

log = logging.getLogger(__name__)
dbg = get_logger("index")

# Paths
KB_DIR = Path("/app/knowledge")
CANON_DIR = Path("/app/canon")
LANCEDB_DIR = Path("/app/lancedb")

# Max tokens for embedding chunks (cl100k_base)
EMBED_MAX_TOKENS = 8000

# Table name
TABLE_NAME = "knowledge_base"

# Medallion tiering (P0.1, B1 adds memory)
# - canon:  gold tier, curated, read-only (canon/*)
# - wiki:   silver tier, agent-compiled, writable (knowledge/* except raw/memory/)
# - memory: between raw and wiki. Distilled notes from past chat threads
#           (knowledge/memory/*). Writable, mid search priority.
# - raw:    bronze tier, source material, read-only, lowest search priority
#           (knowledge/raw/*)
TIER_CANON = "canon"
TIER_WIKI = "wiki"
TIER_MEMORY = "memory"
TIER_RAW = "raw"
RAW_SUBDIR = "raw"
MEMORY_SUBDIR = "memory"

# P0.3 + B1 tier-weighted search ranking. canon is boosted, raw is suppressed,
# memory sits between wiki and raw. The weight multiplies the similarity
# score (1 - cosine_distance) before sorting.
TIER_SEARCH_WEIGHTS = {
    TIER_CANON: 1.15,
    TIER_WIKI: 1.00,
    TIER_MEMORY: 0.90,
    TIER_RAW: 0.75,
}

# Summary model defaults (P0.0 — Ollama-cloud-first with local fallback)
# SUMMARY_PROVIDER controls primary model selection; SUMMARY_MODEL and
# SUMMARY_MODEL_FALLBACK can override either side. Embeddings stay on Gemini
# regardless — these envs only affect the LLM summary calls.
DEFAULT_CLOUD_SUMMARY_MODEL = "qwen3.5:397b-cloud"
DEFAULT_LOCAL_SUMMARY_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434"


def _resolve_summary_config() -> tuple[str, str, str, int]:
    """Resolve summary provider, primary model, fallback model, and worker count.

    Returns (provider, primary_model, fallback_model, max_workers). Provider is
    normalized to either 'cloud_ollama' (default) or 'local_ollama'. Both providers
    talk to the same Ollama /api/chat endpoint; only the model string differs.

    Concurrency: 8 workers for cloud (user-confirmed limit), 3 for local
    (M2 Ultra inference contention).
    """
    provider = os.environ.get("SUMMARY_PROVIDER", "cloud_ollama").lower().strip()
    if provider == "local_ollama":
        primary = os.environ.get("SUMMARY_MODEL", DEFAULT_LOCAL_SUMMARY_MODEL)
        fallback = os.environ.get("SUMMARY_MODEL_FALLBACK", DEFAULT_LOCAL_SUMMARY_MODEL)
        workers = 3
    else:
        provider = "cloud_ollama"
        primary = os.environ.get("SUMMARY_MODEL", DEFAULT_CLOUD_SUMMARY_MODEL)
        fallback = os.environ.get("SUMMARY_MODEL_FALLBACK", DEFAULT_LOCAL_SUMMARY_MODEL)
        workers = 8
    return (provider, primary, fallback, workers)


def _ollama_summary_call(base_url: str, model: str, prompt: str,
                         timeout: float = 120.0) -> str:
    """POST a single summary prompt to Ollama /api/chat. Returns raw content text."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")


def _call_summary_with_fallback(base_url: str, primary: str, fallback: str,
                                prompt: str, log_label: str = "") -> tuple[str, str]:
    """Call primary model with retries; on terminal failure, try fallback once.

    Returns (summary_text, used_model). summary_text=='' means complete failure
    (caller should use mechanical fallback). Retries primary up to 2 extra times
    on 503/429 (so 3 attempts total) with 1s/2s/4s backoff. Non-retryable errors
    skip to fallback immediately.
    """
    import time
    max_retries = 2
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return (_ollama_summary_call(base_url, primary, prompt), primary)
        except Exception as e:
            last_err = e
            err_str = str(e)
            retryable = any(s in err_str for s in ("503", "429", "rate"))
            if retryable and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            break

    if fallback and fallback != primary:
        try:
            text = _ollama_summary_call(base_url, fallback, prompt)
            log.warning(
                f"summary primary={primary} failed ({last_err}); "
                f"fallback={fallback} succeeded {log_label}".strip()
            )
            return (text, fallback)
        except Exception as e:
            log.warning(
                f"summary fallback={fallback} also failed {log_label}: {e}".strip()
            )

    log.warning(f"summary all attempts failed (primary={primary}, fallback={fallback}) {log_label}".strip())
    return ("", "")


def _clean_summary_text(summary: str) -> str:
    """Strip model-specific formatting tags and planning preambles."""
    import re
    if not summary:
        return summary
    summary = re.sub(r'<\|?\w+\|?>[^<]*<\|?\w+\|?>', '', summary).strip()
    summary = re.sub(r'<\|[^>]+\|>', '', summary).strip()
    summary = re.sub(r'^\s*\*?\*?Plan:?\*?\*?\s*\n?', '', summary, flags=re.IGNORECASE)
    summary = re.sub(r'^\s*\d+\.\s+.*?\n', '', summary, count=3)
    if summary.lower().startswith(('plan', '1.', '**plan')):
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        if len(sentences) > 2:
            summary = ' '.join(sentences[-3:])
    return summary.strip()


class GeminiEmbeddingFunction:
    """Embedding function backed by Google Gemini API.

    Uses gemini-embedding-001 with output_dimension=768.
    Fast cloud API — no local model loading.
    Uses the google-genai SDK (current as of 2026).
    """

    def __init__(self, api_key: str, model: str = "gemini-embedding-001",
                 output_dimension: int = 768):
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.model = model
        self.output_dimension = output_dimension

    def name(self) -> str:
        return f"gemini_{self.model}_{self.output_dimension}d"

    def get_config(self) -> dict:
        return {"model": self.model, "output_dimension": self.output_dimension}

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a batch of strings via Gemini API.

        Sends texts in batches of up to 100 (Gemini API limit).
        Falls back to one-at-a-time on batch failure.
        """
        MAX_CHARS = 8000
        MAX_BATCH = 100
        truncated = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in input]

        config = self._types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self.output_dimension,
        )

        # Process in batches of MAX_BATCH
        all_embeddings: list[list[float]] = []
        for i in range(0, len(truncated), MAX_BATCH):
            batch = truncated[i:i + MAX_BATCH]
            try:
                result = self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=config,
                )
                if result.embeddings and len(result.embeddings) == len(batch):
                    all_embeddings.extend([e.values for e in result.embeddings])
                    continue
            except Exception as e:
                log.warning(f"Gemini embed batch failed ({len(batch)} texts): {e}")

            # Fallback: embed one at a time
            for text in batch:
                try:
                    result = self._client.models.embed_content(
                        model=self.model,
                        contents=[text],
                        config=config,
                    )
                    if result.embeddings:
                        all_embeddings.append(result.embeddings[0].values)
                    else:
                        all_embeddings.append([0.0] * self.output_dimension)
                except Exception as e:
                    log.warning(f"Gemini embed single failed (len={len(text)}): {e}")
                    all_embeddings.append([0.0] * self.output_dimension)
        return all_embeddings


class OllamaEmbeddingFunction:
    """Embedding function backed by Ollama /api/embed endpoint."""

    def __init__(self, model: str = "nomic-embed-text",
                 url: str = "http://host.docker.internal:11434",
                 timeout: float = 120.0):
        self.model = model
        self.url = url
        self.timeout = timeout

    def name(self) -> str:
        return f"ollama_{self.model}"

    def get_config(self) -> dict:
        return {"model": self.model, "url": self.url}

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a batch of strings via Ollama.

        Truncates text to fit the model's context window before sending.
        nomic-embed-text has 8192 token context; different tokenizers make
        exact counting hard, so we use a conservative char limit.
        """
        # Conservative: ~8000 chars keeps most text under 8192 tokens
        MAX_CHARS = 8000
        truncated = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in input]

        # Try batch first
        try:
            r = httpx.post(
                f"{self.url}/api/embed",
                json={"model": self.model, "input": truncated},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings) == len(input):
                return embeddings
        except Exception as e:
            log.warning(f"Ollama embed batch failed ({len(input)} texts): {e}")

        # Fallback: embed one at a time, halving text if still too long
        embeddings = []
        for text in truncated:
            emb = self._embed_single(text)
            embeddings.append(emb)
        return embeddings

    def _embed_single(self, text: str, _retries: int = 0) -> list[float]:
        """Embed a single text, retrying with halved length on context overflow."""
        try:
            r = httpx.post(
                f"{self.url}/api/embed",
                json={"model": self.model, "input": [text]},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            embs = data.get("embeddings", [])
            if embs:
                return embs[0]
        except Exception as e:
            if _retries < 3 and len(text) > 500:
                # Context overflow: halve and retry
                half = len(text) // 2
                log.warning(f"Ollama embed retry: halving {len(text)} -> {half} chars")
                return self._embed_single(text[:half], _retries + 1)
            log.warning(f"Ollama embed failed after {_retries} retries (len={len(text)}): {e}")
        return [0.0] * 768


class KBIndex:
    """Semantic index for knowledge base files."""

    def __init__(self, model_gateway=None, embedding_model: str = "nomic-embed-text",
                 embedding_provider: str | None = None):
        import threading
        LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(LANCEDB_DIR))
        self.table = None
        self.model_gateway = model_gateway  # For LLM summaries
        self._file_count = 0  # Number of source files (distinct from chunk/vectors)
        self.graph = None  # Knowledge graph (populated after build_index)
        self._build_lock = threading.Lock()  # Prevent concurrent build_index/reindex_file
        # P1-3: ISO-8601 UTC timestamp of the last successful build_index
        # completion. Surfaced via /kb/graph/overview so the UI can show
        # "last indexed: 12 minutes ago" without a separate metadata file.
        self._last_indexed_at: str | None = None

        # Select embedding provider: "gemini" uses Google API, default uses Ollama
        provider = embedding_provider or os.environ.get("EMBEDDING_PROVIDER", "ollama")

        if provider == "gemini":
            api_key = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError("GOOGLE_GEMINI_API_KEY required for Gemini embeddings")
            self._embedding_model = "gemini-embedding-001"
            self._embedding_fn = GeminiEmbeddingFunction(
                api_key=api_key,
                model="gemini-embedding-001",
                output_dimension=768,
            )
            log.info(f"Using Gemini embeddings (768-dim)")
        else:
            self._embedding_model = embedding_model
            self._embedding_fn = OllamaEmbeddingFunction(
                model=embedding_model,
                url=os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            )
            log.info(f"Using Ollama embeddings ({embedding_model})")

    def build_index(self, extract_entities: bool = False, llm_summaries: bool = False,
                    force: bool = False):
        """Build or rebuild the index from markdown files.

        Chunks each file by headings, generates summaries, and upserts into
        LanceDB with rich metadata.

        Args:
            extract_entities: If True, run langextract entity extraction
                after indexing. Default False — entity extraction is slow and
                should be triggered explicitly via /kb/reindex?entities=true.
            llm_summaries: If True, generate LLM summaries at both document level
                (embedded as "Document Overview" chunk) and section level (replacing
                mechanical fallback). Default False — mechanical summaries only.
            force: If True, wipe and rebuild all chunks regardless of mtime.
                Default False — on startup, only re-index changed files.
        """
        with self._build_lock:
            log.info("Building KB index...")
            log_event(dbg, "build_index_start", extract_entities=extract_entities, llm_summaries=llm_summaries)

            # Open or create table
            if force:
                try:
                    self.db.drop_table(TABLE_NAME)
                except Exception:
                    pass
                self.table = None
            if self.table is None:
                try:
                    self.table = self.db.open_table(TABLE_NAME)
                except Exception:
                    # Create with empty schema — will be populated by _index_file
                    empty = pa.table({
                        "id": pa.array([], type=pa.string()),
                        "vector": pa.array([], type=pa.list_(pa.float32(), 768)),
                        "document": pa.array([], type=pa.string()),
                        "filename": pa.array([], type=pa.string()),
                        "source": pa.array([], type=pa.string()),
                        "heading": pa.array([], type=pa.string()),
                        "chunk_index": pa.array([], type=pa.int64()),
                        "summary": pa.array([], type=pa.string()),
                        "token_count": pa.array([], type=pa.int64()),
                        "mtime": pa.array([], type=pa.int64()),
                        "file_tokens": pa.array([], type=pa.int64()),
                        "section_count": pa.array([], type=pa.int64()),
                        "file_outline": pa.array([], type=pa.string()),
                        "path": pa.array([], type=pa.string()),
                        "folder": pa.array([], type=pa.string()),
                        "type": pa.array([], type=pa.string()),
                        "tier": pa.array([], type=pa.string()),
                    })
                    self.table = self.db.create_table(TABLE_NAME, empty)

            # P0.1 schema migration: backfill `tier` column for legacy tables
            # without re-embedding (preserves all Gemini-generated vectors).
            self._migrate_tier_column()

            # Index all markdown files recursively
            # Skip files whose stored mtime matches current mtime (idempotent startup)
            files_indexed = 0
            files_skipped = 0
            seen_files: set[tuple[str, str]] = set()  # (source, rel_path) for orphan cleanup
            for base_dir in [KB_DIR, CANON_DIR]:
                if not base_dir.exists():
                    continue
                for md_file in base_dir.rglob("**/*.md"):
                    source = "canon" if base_dir == CANON_DIR else "knowledge"
                    try:
                        rel_path = str(md_file.relative_to(base_dir))
                    except ValueError:
                        rel_path = md_file.name

                    seen_files.add((source, rel_path))

                    # Check mtime: skip if unchanged since last index
                    try:
                        current_mtime = int(md_file.stat().st_mtime)
                    except OSError:
                        current_mtime = 0

                    if not force and self._file_mtime_unchanged(rel_path, source, current_mtime):
                        files_skipped += 1
                        continue

                    self._index_file(md_file, rel_path, source, llm_summaries=llm_summaries)
                    files_indexed += 1

            # P0.2 orphan cleanup: delete rows whose source file no longer exists
            # on disk (handles file deletes, moves, and renames). Zero-embedding
            # operation — pure metadata delete.
            orphans_purged = self._purge_orphan_rows(seen_files)

            self._file_count = files_indexed + files_skipped
            log.info(
                f"Indexed {files_indexed} files ({files_skipped} unchanged, skipped, "
                f"{orphans_purged} orphan rows purged)"
            )
            log_event(
                dbg, "index_files_done",
                files=files_indexed, skipped=files_skipped, orphans=orphans_purged,
            )

            # Build knowledge graph: always create structural nodes, then edges, optionally extract entities
            self._init_graph_nodes_only()
            self._build_graph_edges()
            # P2.1: explicit wiki-link / markdown-link REFERENCES edges
            self._build_wiki_link_edges()
            # P2.2: heuristic prose-bridge REFERENCES edges
            self._build_prose_bridge_edges()
            if extract_entities:
                self._extract_entities()
                log.info(f"Graph with entities: {self.graph.get_stats() if self.graph else 'no graph'}")
            else:
                log.info(f"Graph structural nodes: {self.graph.get_stats() if self.graph else 'no graph'}")

            # P1-3: stamp last-build time so /kb/graph/overview can render
            # "indexed N minutes ago" without an external metadata file.
            from datetime import datetime, timezone
            self._last_indexed_at = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )

            return files_indexed

    async def build_index_async(self, extract_entities: bool = False, llm_summaries: bool = False,
                                force: bool = False):
        """Async wrapper for build_index. Runs in a thread pool.

        Needed because _generate_summary() uses sync httpx which could
        block the event loop. This wrapper offloads build_index() to a
        thread.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.build_index(extract_entities, llm_summaries, force=force))

    def _purge_orphan_rows(self, seen_files: set[tuple[str, str]]) -> int:
        """Delete LanceDB rows whose source file is no longer on disk.

        Compares the (source, filename) pairs currently in the table against the
        set of files seen during the rglob pass. Anything in the table but not in
        `seen_files` is considered orphaned (deleted, moved, or renamed) and
        removed. Pure metadata operation — no embeddings recomputed.
        """
        if self.table is None:
            return 0
        try:
            # LanceDB's column projection is exposed via search(), not .select() on
            # the table directly. Falling back to to_pandas() also works but pulls
            # the embedding column too.
            rows = self.table.search().select(["filename", "source"]).to_list()
        except Exception as e:
            log.warning(f"orphan-purge: select failed: {e}")
            return 0
        if not rows:
            return 0

        indexed_pairs = {
            (str(r.get("source", "")), str(r.get("filename", "")))
            for r in rows
        }
        orphans = indexed_pairs - seen_files
        if not orphans:
            return 0

        purged = 0
        for source, filename in orphans:
            safe_filename = filename.replace("'", "''")
            safe_source = source.replace("'", "''")
            try:
                self.table.delete(
                    f"filename = '{safe_filename}' AND source = '{safe_source}'"
                )
                purged += 1
                log.info(f"orphan-purge: removed rows for {source}/{filename}")
            except Exception as e:
                log.warning(
                    f"orphan-purge: delete failed for {source}/{filename}: {e}"
                )
        return purged

    def _migrate_tier_column(self) -> None:
        """Add `tier` column to legacy tables that pre-date P0.1.

        Idempotent: no-op if the column already exists. When migration is
        required, reads existing rows, computes tier from (source, filename),
        drops the old table, and recreates it with the new schema. Embeddings
        are preserved verbatim — zero embedding-API calls.
        """
        if self.table is None:
            return
        try:
            field_names = {f.name for f in self.table.schema}
        except Exception as e:
            log.warning(f"tier-migration: schema read failed: {e}")
            return

        if "tier" in field_names:
            return

        log.info("tier-migration: 'tier' column missing; backfilling in place")
        try:
            df = self.table.to_pandas()
        except Exception as e:
            log.warning(f"tier-migration: to_pandas failed, leaving table as-is: {e}")
            return

        if df.empty:
            try:
                self.db.drop_table(TABLE_NAME)
            except Exception:
                pass
            self.table = None
            return

        df["tier"] = [
            self._compute_tier(str(r.get("source", "")), str(r.get("filename", "")))
            for _, r in df.iterrows()
        ]

        try:
            self.db.drop_table(TABLE_NAME)
            self.table = self.db.create_table(TABLE_NAME, df)
            log.info(f"tier-migration: backfilled {len(df)} rows")
        except Exception as e:
            log.error(f"tier-migration: recreate failed: {e}")
            try:
                self.table = self.db.open_table(TABLE_NAME)
            except Exception:
                self.table = None

    def reindex_file(self, file_path: Path):
        """Reindex a single changed file. Preserves all other rows.

        Used by the file watcher to handle individual file changes without
        wiping the entire index (which would lose LLM summaries).
        """
        with self._build_lock:
            # Determine source and rel_path from file_path
            try:
                rel_path = str(file_path.relative_to(CANON_DIR))
                source = "canon"
            except ValueError:
                try:
                    rel_path = str(file_path.relative_to(KB_DIR))
                    source = "knowledge"
                except ValueError:
                    log.warning(f"File {file_path} not in knowledge/ or canon/, skipping")
                    return

            if not file_path.exists() or not file_path.suffix == ".md":
                return

            log.info(f"Reindexing single file: {rel_path} (source={source})")

            if self.table is None:
                self.table = self.db.open_table(TABLE_NAME)

            self._index_file(file_path, rel_path, source, llm_summaries=False)

            # Rebuild graph edges (fast — uses stored embeddings)
            self._init_graph_nodes_only()
            self._build_graph_edges()
            self._build_wiki_link_edges()
            self._build_prose_bridge_edges()

    def _index_file(self, file_path: Path, rel_path: str, source: str = "knowledge",
                    llm_summaries: bool = False):
        """Index a single markdown file with section-based chunking.

        Each file is split into heading-based chunks. When llm_summaries=True,
        section summaries are generated in parallel by the on-device LLM model,
        and a "Document Overview" chunk is added. Falls back to mechanical
        summaries (first 250 chars) for any section that fails.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            log.error(f"Error reading {file_path}")
            return

        # Snapshot existing summaries by heading BEFORE delete so we can preserve
        # any LLM-grade (or human-edited) summaries when reindexing without
        # llm_summaries=True. Mechanical summaries are NEVER allowed to
        # overwrite a higher-quality stored summary — see _is_mechanical_match.
        existing_summaries: dict[str, str] = {}
        try:
            prior_rows = self.table.search().where(
                f"filename = '{rel_path}' AND source = '{source}'"
            ).select(["heading", "summary"]).to_list()
            for r in prior_rows:
                h = r.get("heading", "")
                s = r.get("summary", "")
                if h and s:
                    existing_summaries[h] = s
        except Exception:
            pass

        # Remove old chunks for this file+source
        try:
            self.table.delete(f"filename = '{rel_path}' AND source = '{source}'")
        except Exception:
            pass

        # Chunk file by headings
        chunks = chunk_file(content, rel_path, max_tokens=EMBED_MAX_TOKENS)
        if not chunks:
            return

        # Compute file-level stats
        file_tokens = sum(c["token_count"] for c in chunks)
        section_count = len(chunks)

        # Build compact outline: unique top-level section names
        seen = []
        for c in chunks:
            top = c["heading"].split(" > ")[0].strip()
            if top not in seen and top != "(preamble)":
                seen.append(top)
        file_outline = " | ".join(seen)
        if len(file_outline) > 500:
            file_outline = file_outline[:497] + "..."

        # Get mtime for manifest tracking
        try:
            mtime = int(file_path.stat().st_mtime)
        except OSError:
            mtime = 0

        # Compute folder relative to base dir
        folder = ""
        try:
            base = KB_DIR if source == "knowledge" else CANON_DIR
            folder = str(file_path.parent.relative_to(base))
        except (ValueError, TypeError):
            pass

        # Medallion tier (P0.1) — derived from source + rel_path
        tier = self._compute_tier(source, rel_path)

        documents = []
        rows = []

        # Optional: generate document-level overview chunk
        if llm_summaries:
            doc_summary = self._generate_doc_summary(content, rel_path)
            if doc_summary and doc_summary != "(no summary)":
                documents.append(doc_summary)
                rows.append({
                    "id": str(uuid.uuid4()),
                    "document": doc_summary,
                    "filename": rel_path,
                    "source": source,
                    "heading": "Document Overview",
                    "chunk_index": -1,
                    "summary": doc_summary[:250],
                    "token_count": count_tokens(doc_summary),
                    "mtime": mtime,
                    "file_tokens": file_tokens,
                    "section_count": section_count,
                    "file_outline": file_outline,
                    "path": str(file_path),
                    "folder": folder,
                    "type": "overview",
                    "tier": tier,
                })
                log_event(dbg, "doc_overview_created", file=rel_path, tokens=count_tokens(doc_summary))

        # Generate section-level LLM summaries when enabled
        section_summaries: dict[int, str] = {}
        if llm_summaries:
            section_summaries = self._generate_section_summaries(chunks, rel_path)

        # Section chunks with summaries.
        #
        # Priority order (mechanical NEVER overwrites a higher-quality summary):
        #   1. Fresh LLM summary from this run (when llm_summaries=True)
        #   2. Prior stored summary, IF it was not mechanical for this chunk's content
        #   3. New mechanical summary (only path that can write a mechanical value)
        #
        # The "is the prior summary mechanical?" check is an exact match against
        # what _mechanical_summary(content) would produce right now — structural
        # detection, no metadata column needed, no false positives against real
        # LLM prose because LLM summaries never reproduce the bare first line.
        for i, chunk in enumerate(chunks):
            fresh_llm = section_summaries.get(i)
            prior = existing_summaries.get(chunk["heading"], "")
            mech = self._mechanical_summary(chunk["content"])
            if fresh_llm:
                summary = fresh_llm
            elif prior and prior != mech and prior != "(no summary)":
                # Preserve prior — it's an LLM summary or a human edit, not the
                # mechanical fallback for this content.
                summary = prior
            else:
                summary = mech
            # Prepend context header for embedding search
            ctx_header = _build_context_header(
                filename=chunk.get("filename", rel_path),
                position=chunk.get("chunk_index", 0) + 1,
                total=chunk.get("total_chunks", section_count),
                heading=chunk["heading"],
                doc_summary=chunk.get("doc_summary", ""),
            )
            doc_text = f"{ctx_header}\n{chunk['content']}"
            documents.append(doc_text)
            rows.append({
                "id": str(uuid.uuid4()),
                "document": doc_text,
                "filename": rel_path,
                "source": source,
                "heading": chunk["heading"],
                "chunk_index": chunk.get("chunk_index", 0),
                "summary": summary,
                "token_count": chunk["token_count"],
                "mtime": mtime,
                "file_tokens": file_tokens,
                "section_count": section_count,
                "file_outline": file_outline,
                "path": str(file_path),
                "folder": folder,
                "type": "section",
                "tier": tier,
            })

        try:
            embeddings = self._embedding_fn(documents)
            for row, emb in zip(rows, embeddings):
                row["vector"] = emb
            self.table.add(rows)
        except Exception as e:
            log.warning(f"Failed to index {rel_path}: {e}")

    def _generate_doc_summary(self, content: str, filename: str) -> str:
        """Generate a document-level summary.

        One LLM call per file. Falls back to mechanical summary on terminal failure.
        Routed through the SUMMARY_PROVIDER env (cloud_ollama default, local_ollama
        forced fallback). On primary cloud-model failure, automatically retries
        once against the local SUMMARY_MODEL_FALLBACK before giving up.
        """
        fallback = self._mechanical_summary(content, max_chars=500)

        if not self.model_gateway:
            return fallback

        provider, primary_model, fallback_model, _workers = _resolve_summary_config()
        base_url = getattr(self.model_gateway, "base_url", None) or \
            os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)

        prompt = (
            f"You are writing a document summary for a personal knowledge base.\n"
            f"The KB covers: philosophy (Stoicism, ethics), neuroscience (HRV, DMN, "
            f"neuroplasticity, vagus nerve), psychology (IFS, CBT, behavioral code), "
            f"and daily practice logs.\n\n"
            f"An AI agent reads this summary to understand the document's scope and "
            f"find cross-domain connections. Preserve searchable signal.\n\n"
            f"Format:\n"
            f"Topic: one-sentence thesis statement.\n"
            f"Covers: comma-separated key topics, named entities, and mechanisms.\n"
            f"Key insights: 2-3 bullet points of core takeaways with specific terms.\n\n"
            f"Document: '{filename}'\n\n"
            f"{content[:6000]}\n\n"
            f"Rules: Be factual and dense. Include proper nouns (authors, brain regions, "
            f"frameworks, acronyms). No preamble. No 'this document discusses'. "
            f"Maximum 6 lines. Extract the core insights, not the wrapping."
        )

        result, used_model = _call_summary_with_fallback(
            base_url, primary_model, fallback_model, prompt,
            log_label=f"doc_summary file={filename}"
        )
        if not result:
            log_event(dbg, "doc_summary_failed", file=filename,
                      provider=provider, primary=primary_model, fallback=fallback_model)
            return fallback

        summary = _clean_summary_text(result) or fallback
        if summary and len(summary) > 10:
            log_event(dbg, "doc_summary_generated", file=filename,
                      provider=provider, model=used_model, len=len(summary))
            return summary[:1000]

        return fallback

    def _generate_section_summaries(
        self, chunks: list[dict], filename: str, max_workers: int | None = None,
    ) -> dict[int, str]:
        """Generate dense LLM summaries for sections in parallel.

        Returns dict mapping chunk_index -> summary string.
        Mechanical fallback for any chunk that fails both primary and fallback model.

        Routing controlled by SUMMARY_PROVIDER (default cloud_ollama, alt local_ollama).
        Both providers go through the Ollama /api/chat endpoint; only the model
        string differs. On per-chunk cloud failure, retries the configured fallback
        (default local llama3.1:8b) once before falling back to mechanical.

        Concurrency defaults: 8 workers cloud, 3 workers local.
        """
        if not self.model_gateway:
            return {}

        results: dict[int, str] = {}

        provider, primary_model, fallback_model, default_workers = _resolve_summary_config()
        base_url = getattr(self.model_gateway, "base_url", None) or \
            os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        if max_workers is None:
            max_workers = default_workers

        def _build_prompt(heading: str, content: str) -> str:
            return (
                f"Summarize this KB section for semantic search. Output ONLY the summary — no planning steps, no numbered lists, no 'Plan' headers.\n\n"
                f"Context: KB covers philosophy (Stoicism, ethics), neuroscience (HRV, DMN, "
                f"neuroplasticity, vagus nerve), psychology (IFS, CBT, behavioral code), "
                f"and daily practice logs.\n\n"
                f"An AI agent reads these summaries to decide whether to load the full section "
                f"and to find cross-domain connections (e.g., Stoic impression → DMN regulation "
                f"→ IFS unblending). Preserve search-relevant signal.\n\n"
                f"Rules:\n"
                f"- 2-3 dense sentences, max 80 words\n"
                f"- Preserve proper nouns: brain regions, author names, frameworks, acronyms\n"
                f"- Include mechanism names: HPA axis, default mode network, vagal tone, etc.\n"
                f"- Note cross-domain bridges when present\n"
                f"- No preamble, no 'this section discusses', no filler phrases\n"
                f"- Do NOT output your planning process — only the final summary\n\n"
                f"Section: '{heading}'\n\n{content}"
            )

        def _summarize_one(idx: int, chunk: dict) -> tuple[int, str]:
            content = chunk["content"][:3000]
            heading = chunk.get("heading", "Unknown Section")
            mech_fallback = self._mechanical_summary(chunk["content"])
            prompt = _build_prompt(heading, content)

            result, used_model = _call_summary_with_fallback(
                base_url, primary_model, fallback_model, prompt,
                log_label=f"section file={filename} heading={heading[:40]}"
            )
            if not result:
                return (idx, mech_fallback)

            summary = _clean_summary_text(result) or mech_fallback
            if summary and len(summary) > 10:
                log_event(dbg, "section_summary_generated",
                          file=filename, heading=heading,
                          provider=provider, model=used_model, len=len(summary))
                return (idx, summary[:500])
            return (idx, mech_fallback)

        log.info(
            f"Generating {len(chunks)} section summaries for {filename} via "
            f"{provider} (primary={primary_model}, fallback={fallback_model}, workers={max_workers})"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_summarize_one, i, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx, summary = future.result()
                results[idx] = summary

        log.info(
            f"Generated {len(results)}/{len(chunks)} section summaries for {filename} via {provider}"
        )
        return results

    @staticmethod
    def _compute_tier(source: str, rel_path: str) -> str:
        """Compute medallion tier for a file.

        - source='canon'                          → 'canon'  (gold)
        - rel_path under knowledge/raw/           → 'raw'    (bronze)
        - rel_path under knowledge/memory/        → 'memory' (between raw/wiki)
        - everything else under knowledge         → 'wiki'   (silver)
        """
        if source == "canon":
            return TIER_CANON
        norm = rel_path.replace("\\", "/").lstrip("/")
        if norm == RAW_SUBDIR or norm.startswith(RAW_SUBDIR + "/"):
            return TIER_RAW
        if norm == MEMORY_SUBDIR or norm.startswith(MEMORY_SUBDIR + "/"):
            return TIER_MEMORY
        return TIER_WIKI

    @staticmethod
    def _mechanical_summary(content: str, max_chars: int = 250) -> str:
        """Extract the first meaningful line as a summary. Used for section-level summaries."""
        for line in content.strip().split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped == "---":
                continue
            if stripped.startswith("[") and "](#" in stripped:
                continue
            if stripped.startswith("|"):
                continue
            if len(stripped) > max_chars:
                return stripped[:max_chars] + "..."
            return stripped
        return "(no summary)"

    def _extract_entities(self) -> None:
        """Extract entities and relationships from indexed chunks and add to graph.

        Assumes _init_graph_nodes_only() has already been called, so CHUNK and
        FOLDER nodes already exist. This method adds ENTITY/CONCEPT nodes,
        REFERENCES edges, and RELATES_TO edges (from relationship extraction).
        """
        from knowledge.graph import Node, NodeType, Edge, EdgeType, extract_entities

        if not self.graph or not self.table:
            return

        # Get all chunks from table (need documents for extraction)
        try:
            df = self.table.to_pandas()
        except Exception as e:
            log.warning(f"Failed to fetch chunks for entity extraction: {e}")
            return

        if df.empty:
            return

        # Extract entities from chunk content (limit to avoid slow API calls)
        model_id = os.environ.get("OLLAMA_MODEL", "glm-5.1:cloud")
        model_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

        # Build name→node index for resolving relationship subjects/objects
        entity_name_index: dict[str, str] = {}  # lowercase name → node id

        for i, row in df.head(200).iterrows():
            chunk_id = row["id"]
            doc = row.get("document", "")

            if not doc:
                continue

            entities = extract_entities(
                content=doc,
                heading=row.get("heading", ""),
                model_id=model_id,
                model_url=model_url,
            )

            # Separate entities from relationships
            relationships = [e for e in entities if e["class"] == "relationship"]
            plain_entities = [e for e in entities if e["class"] != "relationship"]

            # Add entity/concept nodes
            for ent in plain_entities:
                entity_node = Node(
                    id=f"ent_{uuid.uuid4().hex[:8]}",
                    node_type=NodeType.ENTITY if ent["class"] != "concept" else NodeType.CONCEPT,
                    name=ent["text"],
                    source_chunk_id=chunk_id,
                    attributes={"extraction_class": ent["class"], **ent.get("attributes", {})},
                    tags=[row.get("filename", "")],
                )
                merged = self.graph.add_node(entity_node)
                entity_name_index[merged.name.lower()] = merged.id

                # Link entity to its source chunk
                self.graph.add_edge(Edge(
                    source_id=merged.id,
                    target_id=chunk_id,
                    edge_type=EdgeType.REFERENCES,
                    weight=0.9,
                    evidence=f"entity '{ent['text']}' found in chunk",
                ))

            # Create RELATES_TO edges from relationships
            for rel in relationships:
                attrs = rel.get("attributes", {})
                subject = attrs.get("subject", "").lower().strip()
                obj = attrs.get("object", "").lower().strip()
                context = attrs.get("context", "")
                verb = rel.get("text", "")

                if not subject or not obj:
                    continue

                # Resolve subject and object to entity nodes (fuzzy: lowercase substring match)
                subject_id = entity_name_index.get(subject)
                if not subject_id:
                    for name, nid in entity_name_index.items():
                        if subject in name or name in subject:
                            subject_id = nid
                            break
                object_id = entity_name_index.get(obj)
                if not object_id:
                    for name, nid in entity_name_index.items():
                        if obj in name or name in obj:
                            object_id = nid
                            break

                # Create edge between resolved entities, or stash as chunk-level edge
                if subject_id and object_id and subject_id != object_id:
                    evidence = f"{subject} {verb} {obj}"
                    if context:
                        evidence += f" ({context})"
                    self.graph.add_edge(Edge(
                        source_id=subject_id,
                        target_id=object_id,
                        edge_type=EdgeType.RELATES_TO,
                        weight=0.7,
                        evidence=evidence[:200],
                    ))
                else:
                    # Unresolved: link the relationship text to the source chunk
                    self.graph.add_edge(Edge(
                        source_id=chunk_id,
                        target_id=chunk_id,  # self-loop as placeholder
                        edge_type=EdgeType.RELATES_TO,
                        weight=0.3,
                        evidence=f"relationship: {subject} {verb} {obj}"[:200],
                    ))

        self.graph.save()

    def _init_graph_nodes_only(self) -> None:
        """Create graph nodes from table chunks without entity extraction.

        Used in tests and when Ollama is unavailable. Creates chunk-level
        nodes, folder nodes, and folder hierarchy edges.
        """
        from knowledge.graph import (
            KnowledgeGraph, Node, Edge, EdgeType, build_folder_tree,
        )

        self.graph = KnowledgeGraph(LANCEDB_DIR / "graph.json")
        self.graph.clear()  # Full rebuild — discard stale data from previous runs

        if not self.table:
            return

        try:
            df = self.table.to_pandas()
        except Exception as e:
            log.warning(f"Failed to fetch chunks for graph init: {e}")
            return

        if df.empty:
            return

        # Build folder tree from file paths
        folder_data = build_folder_tree(KB_DIR, CANON_DIR)
        for folder_node, edge in folder_data:
            if folder_node:
                self.graph.add_node(folder_node)
            if edge:
                self.graph.add_edge(edge)

        # Create chunk nodes and link to their folder
        for i, row in df.iterrows():
            chunk_id = row["id"]
            source = row.get("source", "knowledge")
            folder = row.get("folder", "")
            # Tier on the row may be missing for legacy rows that survived migration
            # in degraded mode — recompute as a defensive fallback.
            row_tier = row.get("tier") if "tier" in df.columns else None
            tier = (
                str(row_tier)
                if row_tier
                else self._compute_tier(str(source), str(row.get("filename", "")))
            )
            chunk_node = Node(
                id=chunk_id,
                node_type=NodeType.CHUNK,
                name=f"{row.get('filename', '')} > {row.get('heading', '')}",
                filename=row.get("filename", ""),
                heading=row.get("heading", ""),
                summary=row.get("summary", ""),
                attributes={"source": source, "folder": folder, "tier": tier},
                tags=self._extract_tags(row.to_dict()),
            )
            self.graph.add_node(chunk_node)

            # Link chunk to its folder
            if folder:
                folder_id = f"folder_{source}_{folder.replace('/', '_')}"
                folder_node = self.graph.get_node(folder_id)
                if folder_node:
                    self.graph.add_edge(Edge(
                        source_id=chunk_id,
                        target_id=folder_id,
                        edge_type=EdgeType.PARENT_CHILD,
                        weight=1.0,
                        evidence="chunk in folder",
                    ))

        self.graph.save()

    def _build_wiki_link_edges(self) -> None:
        """P2.1: parse [[wiki links]] and [text](file.md#sec) from indexed
        chunks, resolve them against the index, and emit REFERENCES edges
        with link_text + anchor provenance attached.

        Idempotent: re-running on the same content recreates the same edges
        keyed by (source_chunk, target_chunk, REFERENCES) — duplicate adds
        are deduped by KnowledgeGraph.add_edge.
        """
        from knowledge.graph import Edge, EdgeType, NodeType
        from knowledge.wiki_links import (
            parse_links, resolve_link, normalize_anchor,
        )

        if not self.graph or not self.table:
            return
        try:
            df = self.table.to_pandas()
        except Exception as e:
            log.warning(f"wiki-link edge build: failed to fetch chunks: {e}")
            return
        if df.empty:
            return

        try:
            indexed_files = self.list_indexed_filenames()
        except Exception:
            indexed_files = []
        if not indexed_files:
            return

        chunks_by_file: dict[tuple[str, str], list[dict]] = {}
        for nid, node in self.graph.nodes.items():
            if node.node_type != NodeType.CHUNK:
                continue
            key = (node.filename, node.attributes.get("source", "knowledge"))
            chunks_by_file.setdefault(key, []).append({
                "id": nid,
                "heading": node.heading or "",
                "heading_norm": normalize_anchor(node.heading or ""),
                "leaf_norm": normalize_anchor(
                    (node.heading or "").split(" > ")[-1]
                ),
            })

        edges_added = 0
        for _, row in df.iterrows():
            chunk_id = row.get("id")
            doc = row.get("document", "") or ""
            if not chunk_id or not doc:
                continue
            try:
                links = parse_links(doc)
            except Exception:
                continue
            if not links:
                continue

            for link in links:
                resolved = resolve_link(link, indexed_files)
                if not resolved:
                    continue
                tgt_file, tgt_source = resolved
                candidates = chunks_by_file.get((tgt_file, tgt_source), [])
                if not candidates:
                    continue
                anchor_norm = normalize_anchor(link.get("anchor") or "")
                target_chunk_id: str | None = None
                if anchor_norm:
                    for c in candidates:
                        if (
                            c["heading_norm"] == anchor_norm
                            or c["leaf_norm"] == anchor_norm
                            or anchor_norm in c["heading_norm"]
                        ):
                            target_chunk_id = c["id"]
                            break
                if not target_chunk_id:
                    target_chunk_id = candidates[0]["id"]
                if target_chunk_id == chunk_id:
                    continue

                attrs = {
                    "link_text": link.get("display") or link.get("target", ""),
                    "link_kind": link.get("kind", ""),
                    "target_file": tgt_file,
                    "target_anchor": link.get("anchor", "") or "",
                }
                self.graph.add_edge(Edge(
                    source_id=chunk_id,
                    target_id=target_chunk_id,
                    edge_type=EdgeType.REFERENCES,
                    weight=1.0,
                    evidence=f"wiki link: {link.get('raw', '')}",
                    attributes=attrs,
                ))
                edges_added += 1

        if edges_added:
            log.info(f"wiki-link edges added: {edges_added}")
            try:
                self.graph.save()
            except Exception:
                pass

    def _build_prose_bridge_edges(self) -> None:
        """P2.2: heuristic prose-pattern bridge edges.

        For every indexed chunk, scan its body for sentences that mention
        two distinct known KB pages and emit a soft REFERENCES edge with
        link_kind="prose" plus the originating sentence as `evidence`.
        Conservative: see prose_bridges.find_bridges for exact rules.

        Bridge edges go from the source chunk to the target file's first
        chunk (no anchor inference here — prose doesn't carry one). The
        graph dedupes on (source_id, target_id, REFERENCES) so wiki-link
        edges from P2.1 win precedence and prose bridges are additive
        only where no explicit link already exists.
        """
        from knowledge.graph import Edge, EdgeType, NodeType
        from knowledge.prose_bridges import compile_page_index, find_bridges

        if not self.graph or not self.table:
            return
        try:
            df = self.table.to_pandas()
        except Exception as e:
            log.warning(f"prose-bridge build: failed to fetch chunks: {e}")
            return
        if df.empty:
            return

        try:
            indexed_files = self.list_indexed_filenames()
        except Exception:
            indexed_files = []
        if len(indexed_files) < 2:
            return

        # Gather H1 aliases per page so prose mentioning the human heading
        # ("Marcus Aurelius") matches the slug-based file (marcus-aurelius.md).
        aliases_by_key: dict[tuple[str, str], list[str]] = {}
        for nid, node in self.graph.nodes.items():
            if node.node_type != NodeType.CHUNK:
                continue
            heading = (node.heading or "").split(" > ")[0].strip()
            if not heading:
                continue
            key = (node.filename, node.attributes.get("source", "knowledge"))
            aliases_by_key.setdefault(key, []).append(heading)

        known_pages: list[dict] = []
        for f in indexed_files:
            key = (f.get("filename", ""), f.get("source", "knowledge"))
            known_pages.append({
                "filename": key[0],
                "source": key[1],
                "aliases": list(set(aliases_by_key.get(key, []))),
            })

        # Compile the combined page-mention regex ONCE, reuse per chunk.
        compiled = compile_page_index(known_pages)
        if compiled[0] is None:
            return

        # First-chunk lookup per (filename, source) for bridge target resolution
        first_chunk_by_key: dict[tuple[str, str], str] = {}
        for nid, node in self.graph.nodes.items():
            if node.node_type != NodeType.CHUNK:
                continue
            key = (node.filename, node.attributes.get("source", "knowledge"))
            if key not in first_chunk_by_key:
                first_chunk_by_key[key] = nid

        edges_added = 0
        for _, row in df.iterrows():
            chunk_id = row.get("id")
            doc = row.get("document", "") or ""
            if not chunk_id or not doc:
                continue
            try:
                bridges = find_bridges(doc, known_pages, compiled=compiled)
            except Exception:
                continue
            if not bridges:
                continue

            chunk_filename = row.get("filename", "")
            chunk_source = row.get("source", "knowledge")
            self_key = (chunk_filename, chunk_source)

            for b in bridges:
                # Anchor the edge to the source chunk regardless of which
                # half of the sentence the chunk's own page sits on. If
                # neither side is the chunk's own page, the bridge is
                # third-party prose — skip it; we only want bridges that
                # involve the chunk we're scanning.
                subj_key = (b["subject_file"], b["subject_source"])
                obj_key = (b["object_file"], b["object_source"])
                if self_key == subj_key:
                    target_key = obj_key
                    other_match = b["object_match"]
                elif self_key == obj_key:
                    target_key = subj_key
                    other_match = b["subject_match"]
                else:
                    # Third-party prose bridge — both ends external to this
                    # chunk's own page. Skip; lint pass (P3.2) can flag.
                    continue

                target_chunk_id = first_chunk_by_key.get(target_key)
                if not target_chunk_id or target_chunk_id == chunk_id:
                    continue

                attrs = {
                    "link_text": other_match,
                    "link_kind": "prose",
                    "target_file": target_key[0],
                    "target_anchor": "",
                }
                self.graph.add_edge(Edge(
                    source_id=chunk_id,
                    target_id=target_chunk_id,
                    edge_type=EdgeType.REFERENCES,
                    weight=0.5,  # softer than explicit wiki links (1.0)
                    evidence=b["evidence"][:500],
                    attributes=attrs,
                ))
                edges_added += 1

        if edges_added:
            log.info(f"prose-bridge edges added: {edges_added}")
            try:
                self.graph.save()
            except Exception:
                pass

    # D3: Intra-file SIMILAR is the dominant edge type by count. On
    # densely-curated files (e.g. a quote vault with 150+ author sections)
    # cosine scores cluster tight (0.78-0.87) and the long tail is noise.
    # Cap per-node SIMILAR fan-out and tag each edge with its rank so the
    # agent can read "rank 1/5 in file" instead of guessing whether 0.84
    # is signal or floor.
    INTRA_FILE_SIMILAR_THRESHOLD = 0.78  # bumped from 0.75 to lift the floor
    INTRA_FILE_SIMILAR_CAP = 5  # per-node max SIMILAR edges (was unbounded within top_k=12)

    def _build_graph_edges(self) -> None:
        """Compute SIMILAR, INTER_FILE, CROSS_DOMAIN, and heading PARENT_CHILD edges.

        Three-branch similarity logic:
        - SIMILAR: cosine > INTRA_FILE_SIMILAR_THRESHOLD, same file
          (intra-file structural similarity). Capped at INTRA_FILE_SIMILAR_CAP
          per node, ranked 1..N by weight via ``attributes["intra_rank"]``.
        - INTER_FILE: cosine > 0.55, same source, different file (cross-topic connections)
        - CROSS_DOMAIN: cosine > 0.60, different source (knowledge <-> canon)

        Plus heading-name cross-file matching (zero-cost second pass).
        Uses batch cosine similarity on existing embeddings. Zero extra API calls.
        """
        if not self.table or not self.graph:
            return

        # 1. SIMILAR, INTER_FILE, and CROSS_DOMAIN edges via batch cosine similarity
        chunk_nodes = {nid: n for nid, n in self.graph.nodes.items()
                       if n.node_type == NodeType.CHUNK}
        if not chunk_nodes:
            return

        log.info(f"Building edges for {len(chunk_nodes)} chunk nodes (batch cosine)...")

        # Fetch all embeddings + metadata from table
        # NOTE: table.search().where() does a VECTOR search (not SQL SELECT).
        # It returns nearest neighbors to a zero vector, not all matching rows.
        # Use to_pandas() to fetch all rows, then filter in Python.
        try:
            chunk_ids = set(chunk_nodes.keys())
            df = self.table.to_pandas()
            # Filter to only chunk nodes that exist in the graph
            df = df[df["id"].isin(chunk_ids)].reset_index(drop=True)
        except Exception as e:
            log.warning(f"Failed to fetch embeddings for graph edges: {e}")
            return

        if df.empty or "vector" not in df.columns:
            log.warning("No embeddings returned for graph edges")
            return

        ids = df["id"].tolist()
        embs = np.array(df["vector"].tolist(), dtype=np.float32)

        # Normalize + cosine similarity matrix
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embs / norms
        sim = normed @ normed.T  # (N, N) cosine similarity

        top_k = 12
        intra_threshold = self.INTRA_FILE_SIMILAR_THRESHOLD
        intra_cap = self.INTRA_FILE_SIMILAR_CAP
        for i, chunk_id in enumerate(ids):
            chunk_node = chunk_nodes.get(chunk_id)
            if not chunk_node:
                continue
            source = chunk_node.attributes.get("source", "")
            filename = chunk_node.filename or ""
            # top-k neighbors (excluding self)
            scores = sim[i].copy()
            scores[i] = -1.0
            top_idx = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]

            # Collect intra-file SIMILAR candidates separately so we can
            # rank-and-cap them. INTER_FILE and CROSS_DOMAIN emit inline as
            # before — they're already naturally bounded by top_k=12 across
            # the rest of the file.
            similar_candidates: list[tuple[float, str]] = []
            for j in top_idx:
                score = float(scores[j])
                if score <= 0:
                    continue
                sim_id = ids[j]
                sim_source = df.iloc[j]["source"] if j < len(df) else ""
                sim_filename = df.iloc[j]["filename"] if "filename" in df.columns and j < len(df) else ""
                # Ensure string types (pandas may return NaN for missing values)
                sim_source = str(sim_source) if sim_source else ""
                sim_filename = str(sim_filename) if sim_filename else ""
                if score > intra_threshold and source == sim_source and filename == sim_filename:
                    # Same file: intra-file structural similarity — rank+cap
                    # below instead of emitting now.
                    similar_candidates.append((score, sim_id))
                elif score > 0.55 and source == sim_source and filename != sim_filename:
                    # Same source, different file: cross-topic connection
                    self.graph.add_edge(Edge(
                        source_id=chunk_id, target_id=sim_id,
                        edge_type=EdgeType.INTER_FILE,
                        weight=round(score, 3),
                        evidence=f"cross-file ({filename} ↔ {sim_filename}, sim={score:.2f})",
                    ))
                elif score > 0.60 and source != sim_source:
                    self.graph.add_edge(Edge(
                        source_id=chunk_id, target_id=sim_id,
                        edge_type=EdgeType.CROSS_DOMAIN,
                        weight=round(score, 3),
                        evidence=f"cross-domain ({source}→{sim_source})",
                    ))

            # Rank + cap intra-file SIMILAR candidates for this node.
            similar_candidates.sort(key=lambda sc: -sc[0])
            kept = similar_candidates[:intra_cap]
            kept_total = len(kept)
            for rank, (score, sim_id) in enumerate(kept, start=1):
                self.graph.add_edge(Edge(
                    source_id=chunk_id, target_id=sim_id,
                    edge_type=EdgeType.SIMILAR,
                    weight=round(score, 3),
                    evidence=f"semantic similarity ({score:.2f})",
                    attributes={
                        "intra_rank": rank,
                        "intra_total": kept_total,
                    },
                ))

        # 2. Heading-name cross-file matching (zero-cost second pass)
        heading_index: dict[str, list] = {}
        MIN_FILES_FOR_MATCH = 2
        MAX_FILES_FOR_HEADING = 3  # skip generic headings in >3 files
        for nid, node in self.graph.nodes.items():
            if node.node_type != NodeType.CHUNK:
                continue
            if not node.heading:
                continue
            leaf = node.heading.split(" > ")[-1].lower().strip()
            if not leaf:
                continue
            heading_index.setdefault(leaf, []).append(node)

        seen_pairs = set()
        for leaf, nodes in heading_index.items():
            if len(nodes) < MIN_FILES_FOR_MATCH:
                continue
            # Count distinct files
            files = set(n.filename for n in nodes if n.filename)
            if len(files) > MAX_FILES_FOR_HEADING:
                continue  # skip generic headings like "Introduction"
            if len(files) < MIN_FILES_FOR_MATCH:
                continue  # all same file, no cross-file match
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i + 1:]:
                    if n1.filename == n2.filename:
                        continue  # same file, skip
                    pair_key = tuple(sorted([n1.id, n2.id]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    self.graph.add_edge(Edge(
                        source_id=n1.id, target_id=n2.id,
                        edge_type=EdgeType.INTER_FILE,
                        weight=0.65,
                        evidence=f"heading match: '{leaf}'",
                    ))

        # 3. Heading PARENT_CHILD edges
        for nid, node in self.graph.nodes.items():
            if node.node_type != NodeType.CHUNK:
                continue
            heading = node.heading
            if " > " not in heading:
                continue
            parent_heading = " > ".join(heading.split(" > ")[:-1])
            parent_node = self.graph.find_chunk_node(node.filename, parent_heading)
            if parent_node and parent_node.id != nid:
                self.graph.add_edge(Edge(
                    source_id=parent_node.id, target_id=nid,
                    edge_type=EdgeType.PARENT_CHILD, weight=1.0,
                    evidence=f"heading hierarchy ({parent_heading.split(' > ')[-1]} → {heading.split(' > ')[-1]})",
                ))

        self.graph.save()
        log.info(f"Graph edges built: {self.graph.get_stats()}")

    @staticmethod
    def _extract_tags(meta: dict) -> list[str]:
        """Extract tag-like metadata from chunk metadata."""
        tags = []
        source = meta.get("source")
        folder = meta.get("folder")
        tier = meta.get("tier")
        if source and isinstance(source, str):
            tags.append(source)
        if folder and isinstance(folder, str):
            tags.append(folder)
        if tier and isinstance(tier, str):
            # Namespace tier so it can't collide with a folder named 'wiki'/'raw'.
            tags.append(f"tier:{tier}")
        return tags

    def _file_mtime_unchanged(self, rel_path: str, source: str, current_mtime: int) -> bool:
        """Check if a file's mtime matches what's stored in the index.

        Returns True if the file is already indexed with the same mtime,
        meaning it can be skipped during reindex.
        """
        if not self.table:
            return False
        try:
            results = self.table.search().where(
                f"filename = '{rel_path}' AND source = '{source}'"
            ).select(["mtime"]).limit(1).to_list()
            if not results:
                return False
            stored_mtime = results[0].get("mtime", 0)
            return int(stored_mtime) == current_mtime
        except Exception:
            return False

    def _reinit_table(self):
        """Recreate table after error. Drops and rebuilds from scratch."""
        log.warning("Reinitializing LanceDB table after error")
        log_event(dbg, "lancedb_reinit")
        try:
            self.db.drop_table(TABLE_NAME)
        except Exception:
            pass
        self.table = None
        self.build_index(force=True)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search over KB with medallion-tier-weighted ranking.

        Returns list of dicts with path, content, score, tier, and tier-adjusted
        weighted_score. Results are sorted by weighted_score so canon hits
        outrank equally-similar wiki hits, and raw hits drop below both.

        Compatible with existing API consumers (app.py RAG, /kb/search).
        """
        if not self.table:
            return []

        if not query or not query.strip():
            return []

        # Over-fetch so the tier re-ranking can pull canon hits up from
        # positions 6-15 if the raw similarity sort buried them.
        fetch_k = max(top_k * 3, top_k + 5)
        try:
            query_vec = self._embedding_fn([query])[0]
            results = (
                self.table.search(query_vec)
                .distance_type("cosine")
                .limit(fetch_k)
                .to_list()
            )
        except Exception as e:
            log.warning(f"Search failed: {e}")
            return []

        if not results:
            return []

        formatted = []
        for r in results:
            score = 1 - r.get("_distance", 1.0)
            tier = r.get("tier") or self._compute_tier(
                str(r.get("source", "")), str(r.get("filename", ""))
            )
            weight = TIER_SEARCH_WEIGHTS.get(tier, 1.0)
            formatted.append({
                "path": r.get("path", ""),
                "content": r.get("document", ""),
                "score": score,
                "weighted_score": score * weight,
                "tier": tier,
                "heading": r.get("heading", ""),
                "summary": r.get("summary", ""),
                "filename": r.get("filename", ""),
                "source": r.get("source", ""),
                "token_count": r.get("token_count", 0),
                "chunk_index": r.get("chunk_index", 0),
                "section_count": r.get("section_count", 0),
            })

        formatted.sort(key=lambda d: d["weighted_score"], reverse=True)
        return formatted[:top_k]

    def search_grouped(self, query: str, top_k: int = 10) -> list[dict]:
        """Semantic search grouped by file with medallion-tier-weighted ranking.

        Returns list of dicts, one per file, ordered by best tier-weighted score.
        Each dict contains: filename, source, tier, file_tokens, section_count,
        file_outline, hits: [{heading, summary, chunk_index, distance,
        weighted_score}]
        """
        if not self.table:
            return []

        try:
            total = self.table.count_rows()
        except Exception:
            return []

        if total == 0:
            return []

        try:
            query_vec = self._embedding_fn([query])[0]
            # Over-fetch so canon hits buried below raw can surface.
            fetch_limit = min(max(top_k * 3, top_k + 5), total)
            results = (
                self.table.search(query_vec)
                .distance_type("cosine")
                .limit(fetch_limit)
                .to_list()
            )
        except Exception as e:
            log.warning(f"search_grouped query failed: {e}")
            return []

        # Group raw hits by filename
        by_file: dict[str, dict] = {}
        for r in results:
            fn = r.get("filename", "")
            tier = r.get("tier") or self._compute_tier(
                str(r.get("source", "")), str(fn)
            )
            weight = TIER_SEARCH_WEIGHTS.get(tier, 1.0)
            distance = r.get("_distance", 1.0)
            weighted_score = (1 - distance) * weight

            if fn not in by_file:
                by_file[fn] = {
                    "filename": fn,
                    "source": r.get("source", ""),
                    "tier": tier,
                    "file_tokens": r.get("file_tokens", 0),
                    "section_count": r.get("section_count", 0),
                    "file_outline": r.get("file_outline", ""),
                    "hits": [],
                }
            by_file[fn]["hits"].append({
                "heading": r.get("heading", ""),
                "summary": r.get("summary", ""),
                "chunk_index": r.get("chunk_index", 0),
                "distance": distance,
                "weighted_score": weighted_score,
            })

        # Sort files by their best (highest weighted) hit. Tier weights make
        # equally-similar canon files outrank raw, raw fall below wiki.
        grouped = list(by_file.values())
        grouped.sort(
            key=lambda f: max(h["weighted_score"] for h in f["hits"]),
            reverse=True,
        )
        return grouped[:top_k]

    def get_heading_tree(self, filename: str, source: str = "knowledge") -> str | None:
        """Build and format a heading tree for a file.

        Returns the formatted tree string for LLM consumption, or None
        if the file doesn't exist.
        """
        base_dir = CANON_DIR if source == "canon" else KB_DIR
        file_path = base_dir / filename

        if not file_path.exists():
            # Try the other directory
            other_dir = KB_DIR if source == "canon" else CANON_DIR
            file_path = other_dir / filename
            if not file_path.exists():
                return None

        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        tree = build_heading_tree(content, filename)

        # Enrich with summaries from the index
        summaries = self.get_summaries(filename, source)
        if summaries:
            enrich_tree_summaries(tree, summaries)

        tree_text = format_heading_tree(tree)

        # Prepend Document Overview if it exists in the index
        overview = summaries.get("Document Overview", "")
        if overview:
            tree_text = f"[Document Overview: {overview}]\n{tree_text}"

        return tree_text

    def get_section(self, filename: str, section: str, source: str = "knowledge") -> str | None:
        """Extract a specific section from a file by heading name.

        H1 headings load the entire concept block (all children).
        H2-H5 headings load just that subsection within its parent H1.
        Case-insensitive substring matching.

        Returns the section content, or None if not found.
        """
        base_dir = CANON_DIR if source == "canon" else KB_DIR
        file_path = base_dir / filename

        if not file_path.exists():
            other_dir = KB_DIR if source == "canon" else CANON_DIR
            file_path = other_dir / filename
            if not file_path.exists():
                return None

        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        chunks = chunk_file(content, filename, max_tokens=EMBED_MAX_TOKENS)
        section_lower = section.lower()

        # Pass 1: H1 concept heading match (case-insensitive substring)
        for chunk in chunks:
            if section_lower in chunk["heading"].lower():
                return chunk["content"]

        # Pass 2: H2-H5 subsection extraction
        import re
        heading_re = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
        for chunk in chunks:
            lines = chunk["content"].split("\n")
            for i, line in enumerate(lines):
                m = heading_re.match(line)
                if not m or section_lower not in m.group(2).lower():
                    continue
                level = len(m.group(1))
                extracted = [line]
                for j in range(i + 1, len(lines)):
                    nm = re.match(r"^(#{1,6})\s", lines[j])
                    if nm and len(nm.group(1)) <= level:
                        break
                    extracted.append(lines[j])
                return "\n".join(extracted).strip()

        return None

    def list_indexed_filenames(self) -> list[dict]:
        """Return deduplicated list of files present in the index.

        Each entry: {"filename": rel_path, "source": "knowledge"|"canon",
        "tier": "canon"|"wiki"|"raw"|""}.

        Used by KBTools._resolve_kb_filename for path-strip / substring /
        fuzzy fallbacks so the agent can pass loose filenames like "page.md"
        or "wiki/page.md" and still hit the right file.
        """
        if not self.table:
            return []
        try:
            rows = (
                self.table.search()
                .select(["filename", "source", "tier"])
                .to_list()
            )
        except Exception:
            return []
        seen: set[tuple[str, str]] = set()
        out: list[dict] = []
        for r in rows:
            fn = r.get("filename", "") or ""
            src = r.get("source", "") or ""
            if not fn:
                continue
            key = (fn, src)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "filename": fn,
                "source": src,
                "tier": (r.get("tier", "") or ""),
            })
        return out

    def list_sections(
        self, filename: str, source: str = "knowledge"
    ) -> list[dict]:
        """Return indexed sections for a file ordered by chunk_index.

        Each entry includes heading, chunk_index, token_count, summary, tier.
        Pure index lookup — no disk read, no chunking. Used by
        read_knowledge_section to navigate by indexed metadata before
        falling back to disk for the actual content slice.
        """
        if not self.table:
            return []
        try:
            total = self.table.count_rows()
        except Exception:
            return []
        if total == 0:
            return []

        safe_filename = filename.replace("'", "''")
        safe_source = source.replace("'", "''")
        try:
            rows = (
                self.table.search()
                .where(
                    f"filename = '{safe_filename}' "
                    f"AND source = '{safe_source}' "
                    f"AND type = 'section'"
                )
                .select(["heading", "chunk_index", "token_count", "summary", "tier"])
                .to_list()
            )
        except Exception:
            return []
        rows.sort(key=lambda r: r.get("chunk_index", 0))
        return rows

    def get_summaries(self, filename: str, source: str = "knowledge") -> dict[str, str]:
        """Fetch heading -> summary mapping for a file from the index.

        Returns a dict mapping heading names to their summaries.
        Returns empty dict if file is not indexed.
        """
        if not self.table:
            return {}

        try:
            total = self.table.count_rows()
        except Exception:
            return {}

        if total == 0:
            return {}

        try:
            rows = self.table.search().where(
                f"filename = '{filename}' AND source = '{source}'"
            ).select(["heading", "summary"]).to_list()
        except Exception:
            return {}

        summaries: dict[str, str] = {}
        for r in rows:
            heading = r.get("heading", "")
            summary = r.get("summary", "")
            if heading and summary:
                summaries[heading] = summary
        return summaries

    def get_stats(self) -> dict:
        """Get index statistics."""
        if not self.table:
            return {"files": 0, "vectors": 0}

        try:
            vector_count = self.table.count_rows()
        except Exception:
            vector_count = 0

        stats = {
            "files": self._file_count,
            "vectors": vector_count,
            "embedding_model": self._embedding_model,
        }
        if self.graph:
            stats["graph"] = self.graph.get_stats()
        return stats

    def get_folder_tree(
        self, source: str = "knowledge", root_path: str | None = None
    ) -> str:
        """Get the folder hierarchy tree for LLM consumption.

        Returns formatted tree string showing folder structure with
        file counts, summaries, and subfolder relationships.

        Args:
            source: ``"canon"`` or ``"knowledge"``.
            root_path: Optional sub-path inside ``source`` to re-root the
                tree (e.g. ``"mind-en-place"`` or ``"raw/technology"``).
        """
        from knowledge.graph import format_folder_tree
        if not self.graph:
            return "(No graph available — run build_index first)"
        return format_folder_tree(
            self.graph, source=source, root_path=root_path
        )
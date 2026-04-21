"""
Knowledge Base File Watcher

Uses watchdog to detect file changes in knowledge/ and canon/ directories.
Triggers re-indexing when files change.

Includes a path-suppression hook (`suppress_paths`) that callers can use to
silence watchdog events for files THEY are about to write themselves —
prevents the cascade where save_knowledge writes file.md + log.md + index.md,
each of which would otherwise trigger another reindex on top of the one
save_knowledge already performed inline.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

log = logging.getLogger(__name__)


# Per-process suppression registry (path -> expiry epoch). Anyone (e.g.,
# save_knowledge) can call suppress_paths(...) before a write to tell the
# watcher "I'm the one writing these — don't trigger a redundant reindex."
_suppression_lock = threading.Lock()
_suppressed: dict[str, float] = {}
_DEFAULT_SUPPRESS_SECONDS = 5.0


def suppress_paths(paths: Iterable[Path | str], seconds: float = _DEFAULT_SUPPRESS_SECONDS) -> None:
    """Register paths whose next watchdog event(s) should be ignored.

    Used by save_knowledge to avoid the watcher re-indexing files we just
    indexed inline. Suppression auto-expires after `seconds` so a stale
    entry can never permanently mute a real edit.
    """
    if seconds <= 0:
        return
    expiry = time.monotonic() + seconds
    with _suppression_lock:
        for p in paths:
            try:
                resolved = str(Path(p).resolve())
            except Exception:
                resolved = str(p)
            _suppressed[resolved] = expiry


def _is_suppressed(path: Path) -> bool:
    """Return True if `path` is currently suppressed (and prune expired)."""
    try:
        key = str(path.resolve())
    except Exception:
        key = str(path)
    now = time.monotonic()
    with _suppression_lock:
        # Prune expired entries opportunistically.
        if _suppressed:
            for k in [k for k, exp in _suppressed.items() if exp <= now]:
                _suppressed.pop(k, None)
        return key in _suppressed


def clear_suppressions() -> None:
    """Test helper: drop all suppressions."""
    with _suppression_lock:
        _suppressed.clear()


class KBEventHandler(FileSystemEventHandler):
    """Handle file system events for knowledge base."""

    def __init__(self, reindex_callback: Optional[Callable] = None):
        self.reindex_callback = reindex_callback

    def _trigger_reindex(self, path: Path, event_type: str):
        """Trigger reindex if callback is set. Passes the changed file path.

        Honors suppress_paths() — callers writing files inline can register
        them as suppressed to skip a redundant watcher-driven reindex.
        """
        if _is_suppressed(path):
            log.debug(f"KB file {event_type} (suppressed by caller): {path}")
            return
        log.info(f"KB file {event_type}: {path}")
        if self.reindex_callback:
            log.info(f"Triggering KB reindex for {path}...")
            self.reindex_callback(path)

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent) and event.src_path.endswith(".md"):
            path = Path(event.src_path)
            self._trigger_reindex(path, "modified")

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent) and event.src_path.endswith(".md"):
            path = Path(event.src_path)
            self._trigger_reindex(path, "created")


class KnowledgeBaseWatcher:
    """Watch knowledge base directories for changes."""

    def __init__(self, kb_dir: Path, canon_dir: Path, reindex_callback: Optional[Callable] = None):
        self.kb_dir = kb_dir
        self.canon_dir = canon_dir
        self.reindex_callback = reindex_callback
        self.observer = Observer()

    def start(self):
        """Start watching directories."""
        event_handler = KBEventHandler(self.reindex_callback)

        # Watch both directories recursively (subfolders have semantic meaning)
        self.observer.schedule(event_handler, str(self.kb_dir), recursive=True)
        self.observer.schedule(event_handler, str(self.canon_dir), recursive=True)

        self.observer.start()
        log.info("Knowledge base watcher started")

    def stop(self):
        """Stop watching."""
        self.observer.stop()
        self.observer.join()

"""
Tests for knowledge base file watcher.
"""

import pytest
import time
from pathlib import Path
from agent.watcher import KnowledgeBaseWatcher, KBEventHandler


class TestKnowledgeBaseWatcher:
    """Test file system watcher for KB changes."""

    def test_watcher_init(self, temp_kb_dir):
        """Test watcher initialization."""
        kb_dir, canon_dir = temp_kb_dir
        watcher = KnowledgeBaseWatcher(kb_dir, canon_dir)

        assert watcher.kb_dir == kb_dir
        assert watcher.canon_dir == canon_dir
        assert watcher.observer is not None

    def test_watcher_start_stop(self, temp_kb_dir):
        """Test watcher start and stop."""
        kb_dir, canon_dir = temp_kb_dir
        watcher = KnowledgeBaseWatcher(kb_dir, canon_dir)

        watcher.start()
        time.sleep(0.1)  # Let observer start

        watcher.stop()
        # Should not raise

    def test_event_handler_md_filter(self, temp_kb_dir):
        """Test event handler filters for .md files."""
        kb_dir, canon_dir = temp_kb_dir
        handler = KBEventHandler()

        # Should handle .md files
        from watchdog.events import FileModifiedEvent
        event = FileModifiedEvent(str(kb_dir / "test.md"))
        handler.on_modified(event)
        # Should not raise

        # Non-md files should be ignored (no logging)
        event = FileModifiedEvent(str(kb_dir / "test.txt"))
        handler.on_modified(event)


class TestKBEventHandler:
    """Test KB event handler."""

    def test_on_modified_filters_md(self, temp_kb_dir):
        """Test modified event only logs .md files."""
        kb_dir, canon_dir = temp_kb_dir
        handler = KBEventHandler()

        from watchdog.events import FileModifiedEvent

        # .md file should log
        event = FileModifiedEvent(str(kb_dir / "test.md"))
        handler.on_modified(event)
        # Should not raise

        # Non-md should be ignored
        event = FileModifiedEvent(str(kb_dir / "test.txt"))
        handler.on_modified(event)
        # Should not raise

    def test_on_created_filters_md(self, temp_kb_dir):
        """Test created event only logs .md files."""
        kb_dir, canon_dir = temp_kb_dir
        handler = KBEventHandler()

        from watchdog.events import FileCreatedEvent

        event = FileCreatedEvent(str(kb_dir / "new.md"))
        handler.on_created(event)
        # Should not raise

        event = FileCreatedEvent(str(kb_dir / "new.txt"))
        handler.on_created(event)
        # Should not raise

    def test_callback_receives_file_path(self, temp_kb_dir):
        """Callback receives the changed file's Path, not a bare call."""
        kb_dir, canon_dir = temp_kb_dir
        received_paths = []
        handler = KBEventHandler(reindex_callback=lambda p: received_paths.append(p))

        from watchdog.events import FileModifiedEvent
        test_path = kb_dir / "test.md"
        event = FileModifiedEvent(str(test_path))
        handler.on_modified(event)

        assert len(received_paths) == 1
        assert received_paths[0] == Path(str(test_path))

    def test_callback_receives_path_on_created(self, temp_kb_dir):
        """Callback receives file path on creation events too."""
        kb_dir, canon_dir = temp_kb_dir
        received_paths = []
        handler = KBEventHandler(reindex_callback=lambda p: received_paths.append(p))

        from watchdog.events import FileCreatedEvent
        test_path = kb_dir / "new.md"
        event = FileCreatedEvent(str(test_path))
        handler.on_created(event)

        assert len(received_paths) == 1


class TestSuppressPaths:
    """suppress_paths() lets a caller silence the next watcher event for a
    file it's about to write — used by save_knowledge to avoid the cascade
    where it indexes inline AND the watcher re-indexes on top.
    """

    def setup_method(self):
        from agent.watcher import clear_suppressions
        clear_suppressions()

    def test_suppressed_path_skips_callback(self, temp_kb_dir):
        kb_dir, _ = temp_kb_dir
        target = kb_dir / "x.md"
        target.write_text("# X")

        from agent.watcher import suppress_paths
        from watchdog.events import FileModifiedEvent

        received = []
        handler = KBEventHandler(reindex_callback=lambda p: received.append(p))
        suppress_paths([target])
        handler.on_modified(FileModifiedEvent(str(target)))

        assert received == [], "suppressed path should not trigger callback"

    def test_unsuppressed_path_still_fires(self, temp_kb_dir):
        kb_dir, _ = temp_kb_dir
        suppressed = kb_dir / "a.md"
        other = kb_dir / "b.md"
        suppressed.write_text("# A")
        other.write_text("# B")

        from agent.watcher import suppress_paths
        from watchdog.events import FileModifiedEvent

        received = []
        handler = KBEventHandler(reindex_callback=lambda p: received.append(p))
        suppress_paths([suppressed])
        handler.on_modified(FileModifiedEvent(str(other)))

        assert len(received) == 1, "non-suppressed path must still fire"

    def test_suppression_expires(self, temp_kb_dir):
        """A stale suppression entry can never permanently mute real edits."""
        kb_dir, _ = temp_kb_dir
        target = kb_dir / "x.md"
        target.write_text("# X")

        from agent.watcher import suppress_paths
        from watchdog.events import FileModifiedEvent

        received = []
        handler = KBEventHandler(reindex_callback=lambda p: received.append(p))
        suppress_paths([target], seconds=0.05)

        # Inside the window — suppressed
        handler.on_modified(FileModifiedEvent(str(target)))
        assert received == []

        # Past expiry — not suppressed any more
        time.sleep(0.1)
        handler.on_modified(FileModifiedEvent(str(target)))
        assert len(received) == 1, "expired suppression must release the path"

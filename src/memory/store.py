"""
Conversation Store

JSON file-based conversation session management.
Each conversation is a single JSON file in the sessions directory.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:  # optional: token-aware budget walk; falls back to naive char count
    from agent.tokenizer import count_tokens as _count_tokens
except Exception:  # pragma: no cover -- defensive import
    def _count_tokens(text: str) -> int:
        return max(1, len(text or "") // 4)

log = logging.getLogger(__name__)

SESSIONS_DIR = Path("/app/sessions")

# Reserved metadata keys recognized on a turn.
# tool_calls:    list of {"name": str, "args": dict}
# tool_results:  list of {"name": str, "preview": str, "delivered_chars": int,
#                          "original_chars": int, "truncated": bool}
TURN_METADATA_KEYS = ("tool_calls", "tool_results")


class ConversationStore:
    """Persistent conversation store using JSON files."""

    def __init__(self, sessions_dir: Optional[Path] = None):
        self.sessions_dir = sessions_dir or SESSIONS_DIR

    def initialize(self):
        """Create sessions directory."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Conversation store initialized at {self.sessions_dir}")

    def create_conversation(self) -> str:
        """Create a new conversation. Returns conversation ID."""
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._write_session(conv_id, {
            "id": conv_id,
            "title": "New Chat",
            "created_at": now,
            "updated_at": now,
            "turns": [],
        })
        return conv_id

    def add_turn(self, role: str, content: str, conversation_id: str,
                 metadata: Optional[dict] = None):
        """Add a conversation turn to a session.

        metadata may include reserved keys (see TURN_METADATA_KEYS) which
        are stored as structured fields on the turn rather than buried in
        the prose. This lets future history reconstruction inject compact
        summaries instead of replaying raw tool brackets back to the model.

        Schema per turn: {role, content, timestamp, tool_calls?, tool_results?}
        """
        session = self._read_session(conversation_id)
        now = datetime.now(timezone.utc).isoformat()
        turn = {"role": role, "content": content, "timestamp": now}
        if metadata:
            turn.update(metadata)
        session["turns"].append(turn)
        session["updated_at"] = now
        # Auto-title from first user message
        if role == "user" and session.get("title") == "New Chat":
            title = content[:50] + ("..." if len(content) > 50 else "")
            session["title"] = title
        self._write_session(conversation_id, session)

    def get_recent(self, conversation_id: str, n: int = 10) -> list[dict]:
        """Get the last N turns from a conversation.

        Kept for back-compat. New callers should prefer
        get_history_within_budget which is token-aware.
        """
        session = self._read_session(conversation_id)
        return session.get("turns", [])[-n:]

    def get_history_within_budget(
        self,
        conversation_id: str,
        max_tokens: int,
        always_full_n: int = 2,
    ) -> list[dict]:
        """Walk turns newest→oldest, return as many as fit under max_tokens.

        - Always includes the last `always_full_n` turns in full so the
          immediate context is intact.
        - Older turns with `tool_calls`/`tool_results` metadata are
          rendered as compact stubs (e.g. "[earlier: I loaded sections
          A, B from foo.md and ran graph_stats]") so the model can recall
          what happened without re-replaying entire tool transcripts.
        - Returns turns in chronological order (oldest first), suitable
          for direct use as the LLM messages array prefix.

        max_tokens applies to the cumulative content size; per-message
        overhead is not counted here.
        """
        session = self._read_session(conversation_id)
        turns = session.get("turns", [])
        if not turns:
            return []

        kept_reversed: list[dict] = []
        used = 0

        for idx_from_end, turn in enumerate(reversed(turns)):
            is_recent = idx_from_end < always_full_n
            rendered = turn if is_recent else self._compact_turn(turn)
            cost = _count_tokens(rendered.get("content") or "")
            if not is_recent and used + cost > max_tokens:
                break
            kept_reversed.append(rendered)
            used += cost

        kept_reversed.reverse()
        return kept_reversed

    @staticmethod
    def _compact_turn(turn: dict) -> dict:
        """Render an older turn as a compact summary if it carried tool metadata.

        Assistant turns that ran tools become a single-line stub citing the
        tool names and any sections that were loaded. The original content
        (which may contain raw tool brackets) is dropped on purpose so the
        model doesn't re-emit the same calls.
        """
        tool_calls = turn.get("tool_calls") or []
        tool_results = turn.get("tool_results") or []
        if not tool_calls and not tool_results:
            return {"role": turn.get("role", "assistant"),
                    "content": turn.get("content", "")}

        sections_loaded = []
        files_touched = set()
        tool_names = []
        for tc in tool_calls:
            name = tc.get("name", "?")
            tool_names.append(name)
            args = tc.get("args") or {}
            # Heuristic: read_knowledge_section args are filename/section
            if name == "read_knowledge_section":
                fn = args.get("0") or args.get("filename") or ""
                sec = args.get("1") or args.get("section") or ""
                if fn:
                    files_touched.add(fn)
                if sec:
                    sections_loaded.append(f"{sec} from {fn}" if fn else sec)
            elif name in ("read_knowledge", "search_knowledge"):
                arg0 = args.get("0") or ""
                if arg0:
                    files_touched.add(arg0)

        parts = ["[earlier turn:"]
        if tool_names:
            unique_tools = ", ".join(sorted(set(tool_names)))
            parts.append(f"called {unique_tools}")
        if sections_loaded:
            preview = "; ".join(sections_loaded[:5])
            more = f" (+{len(sections_loaded) - 5} more)" if len(sections_loaded) > 5 else ""
            parts.append(f"; loaded sections: {preview}{more}")
        elif files_touched:
            preview = ", ".join(sorted(files_touched)[:5])
            parts.append(f"; touched files: {preview}")
        parts.append("]")
        return {"role": turn.get("role", "assistant"), "content": " ".join(parts)}

    def list_conversations(self) -> list[dict]:
        """List all conversations, most recently updated first."""
        results = []
        for f in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                results.append({
                    "id": data["id"],
                    "title": data.get("title", "New Chat"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "turn_count": len(data.get("turns", [])),
                })
            except Exception:
                continue
        results.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        return results

    def get_conversation(self, conversation_id: str) -> list[dict]:
        """Get all turns for a conversation."""
        session = self._read_session(conversation_id)
        return session.get("turns", [])

    def delete_conversation(self, conversation_id: str):
        """Delete a conversation."""
        path = self.sessions_dir / f"{conversation_id}.json"
        if path.exists():
            path.unlink()

    def _read_session(self, conversation_id: str) -> dict:
        """Read a session JSON file."""
        path = self.sessions_dir / f"{conversation_id}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {
            "id": conversation_id,
            "title": "New Chat",
            "created_at": "",
            "updated_at": "",
            "turns": [],
        }

    def _write_session(self, conversation_id: str, data: dict):
        """Write a session JSON file."""
        path = self.sessions_dir / f"{conversation_id}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
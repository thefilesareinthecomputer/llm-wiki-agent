"""Lightweight knowledge graph with entity extraction.

Nodes represent chunks (sections) and extracted entities (concepts, people, technologies).
Edges represent semantic similarity, heading hierarchy, and entity relationships.
Persisted to JSON for survival across container restarts.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class EdgeType(Enum):
    SIMILAR = "similar"
    CROSS_DOMAIN = "cross_domain"
    PARENT_CHILD = "parent_child"
    REFERENCES = "references"
    RELATES_TO = "relates_to"
    INTER_FILE = "inter_file"


class NodeType(Enum):
    CHUNK = "chunk"
    ENTITY = "entity"
    CONCEPT = "concept"
    FOLDER = "folder"


@dataclass
class Node:
    id: str
    node_type: NodeType
    name: str
    filename: str = ""
    heading: str = ""
    summary: str = ""
    source_chunk_id: str = ""
    attributes: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        d = dict(d)
        d["node_type"] = NodeType(d["node_type"])
        return cls(**d)


@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 0.0
    evidence: str = ""
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edge_type"] = self.edge_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        d = dict(d)
        d["edge_type"] = EdgeType(d["edge_type"])
        # Backward compat: legacy persisted edges have no `attributes` key.
        d.setdefault("attributes", {})
        return cls(**d)

    @property
    def key(self) -> str:
        return f"{self.source_id}:{self.target_id}:{self.edge_type.value}"


class KnowledgeGraph:
    """In-memory knowledge graph backed by JSON persistence."""

    def __init__(self, persist_path: Path):
        self.persist_path = persist_path
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}  # keyed by edge.key
        self._load()

    def add_node(self, node: Node) -> Node:
        """Add or merge a node. Same name+type merges attributes."""
        # Entity dedup: same name + type → merge
        if node.node_type in (NodeType.ENTITY, NodeType.CONCEPT):
            for existing in self.nodes.values():
                if (existing.node_type == node.node_type
                        and existing.name.lower() == node.name.lower()):
                    # Merge: keep both source references
                    if node.source_chunk_id and node.source_chunk_id not in existing.attributes.get("sources", []):
                        sources = existing.attributes.get("sources", [])
                        if existing.source_chunk_id:
                            sources.append(existing.source_chunk_id)
                        sources.append(node.source_chunk_id)
                        existing.attributes["sources"] = sources
                    # Merge tags
                    for tag in node.tags:
                        if tag not in existing.tags:
                            existing.tags.append(tag)
                    return existing

        if node.id in self.nodes:
            return self.nodes[node.id]
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: Edge) -> None:
        """Add or update an edge. Same key keeps highest weight."""
        key = edge.key
        if key in self.edges:
            if edge.weight > self.edges[key].weight:
                self.edges[key] = edge
        else:
            self.edges[key] = edge

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> list[tuple[Node, Edge]]:
        """Get neighbors of a node, optionally filtered by edge type.

        Dedupes on (neighbor_id, edge_type) keeping the highest-weight edge.
        Without this, A->B and B->A SIMILAR edges (same pair, different
        direction) both surface and the agent sees duplicates.
        """
        best: dict[tuple[str, EdgeType], tuple[Node, Edge]] = {}
        for edge in self.edges.values():
            neighbor_id = None
            if edge.source_id == node_id:
                neighbor_id = edge.target_id
            elif edge.target_id == node_id:
                neighbor_id = edge.source_id

            if neighbor_id is None:
                continue
            if edge_type and edge.edge_type != edge_type:
                continue

            neighbor = self.nodes.get(neighbor_id)
            if not neighbor:
                continue

            key = (neighbor_id, edge.edge_type)
            existing = best.get(key)
            if existing is None or edge.weight > existing[1].weight:
                best[key] = (neighbor, edge)
        return list(best.values())

    def traverse(self, start_id: str, max_depth: int = 2,
                 exclude_edge_types: set[EdgeType] | None = None) -> list[tuple[Node, Edge, int]]:
        """BFS traversal from a node. Returns (node, edge, depth) tuples.

        Args:
            start_id: Node to start from.
            max_depth: Maximum traversal depth.
            exclude_edge_types: Edge types to skip during traversal.
                PARENT_CHILD edges are structural (H1→H2 hierarchy), not semantic —
                including them causes exponential explosion.
        """
        visited = {start_id}
        queue: list[tuple[str, int]] = [(start_id, 0)]
        results = []

        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for neighbor, edge in self.get_neighbors(current_id):
                if neighbor.id in visited:
                    continue
                if exclude_edge_types and edge.edge_type in exclude_edge_types:
                    continue
                visited.add(neighbor.id)
                results.append((neighbor, edge, depth + 1))
                queue.append((neighbor.id, depth + 1))

        return results

    def search_entities(self, query: str, top_k: int = 10) -> list[Node]:
        """Search entity/concept nodes by name (substring match)."""
        query_lower = query.lower()
        matches = []
        for node in self.nodes.values():
            if node.node_type == NodeType.CHUNK:
                continue
            if query_lower in node.name.lower():
                matches.append(node)
        # Sort by tag count (more tagged = more referenced)
        matches.sort(key=lambda n: len(n.tags), reverse=True)
        return matches[:top_k]

    def find_chunk_node(self, filename: str, heading: str) -> Optional[Node]:
        """Find a chunk node by filename and heading."""
        for node in self.nodes.values():
            if node.node_type == NodeType.CHUNK and node.filename == filename and heading.lower() in node.heading.lower():
                return node
        return None

    def get_stats(self) -> dict:
        """Get graph statistics including connectivity metrics."""
        type_counts = {}
        for node in self.nodes.values():
            t = node.node_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        edge_type_counts = {}
        for edge in self.edges.values():
            t = edge.edge_type.value
            edge_type_counts[t] = edge_type_counts.get(t, 0) + 1

        # Compute connectivity metrics (all edges, used for orphan/avg metrics)
        node_edge_counts: dict[str, int] = {}
        for edge in self.edges.values():
            node_edge_counts[edge.source_id] = node_edge_counts.get(edge.source_id, 0) + 1
            node_edge_counts[edge.target_id] = node_edge_counts.get(edge.target_id, 0) + 1

        connected_nodes = set(node_edge_counts.keys())
        orphan_nodes = len(self.nodes) - len(connected_nodes)
        avg_edges = len(self.edges) * 2 / len(self.nodes) if self.nodes else 0

        # Most-connected nodes (top 5) — exclude PARENT_CHILD so structural
        # heading-hierarchy nesting doesn't dominate the leaderboard.
        # Semantic intra-file connectivity is preserved by SIMILAR edges.
        semantic_edge_counts: dict[str, int] = {}
        for edge in self.edges.values():
            if edge.edge_type == EdgeType.PARENT_CHILD:
                continue
            semantic_edge_counts[edge.source_id] = semantic_edge_counts.get(edge.source_id, 0) + 1
            semantic_edge_counts[edge.target_id] = semantic_edge_counts.get(edge.target_id, 0) + 1

        # P0-2: total non-parent_child edges, used for per-hub share %.
        total_non_pc_edges = sum(
            cnt for et, cnt in edge_type_counts.items() if et != "parent_child"
        )

        node_conn = []
        for nid, count in sorted(semantic_edge_counts.items(), key=lambda x: -x[1])[:5]:
            node = self.nodes.get(nid)
            if node:
                share = (count / total_non_pc_edges) if total_non_pc_edges > 0 else 0.0
                node_conn.append({
                    "name": node.name,
                    "filename": node.filename,
                    "heading": node.heading,
                    "count": count,
                    "share_non_pc": round(share, 4),
                })

        # P0-2: edge_share — per-edge-type fraction of total edges. Single
        # canonical numbers the agent and the UI both consume so neither
        # side has to recompute from raw counts. Only includes edge types
        # with count > 0.
        total_edges = len(self.edges)
        edge_share = {}
        if total_edges > 0:
            for et, cnt in edge_type_counts.items():
                if cnt > 0:
                    edge_share[et] = round(cnt / total_edges, 4)

        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "node_types": type_counts,
            "edge_types": edge_type_counts,
            "edge_share": edge_share,
            "orphan_nodes": orphan_nodes,
            "avg_edges_per_node": round(avg_edges, 1),
            "most_connected": node_conn,
        }

    def save(self) -> None:
        """Persist graph to JSON."""
        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
        }
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps(data, indent=2))
            log.info(f"Graph saved: {len(self.nodes)} nodes, {len(self.edges)} edges")
        except Exception as e:
            log.error(f"Failed to save graph: {e}")

    def _load(self) -> None:
        """Load graph from JSON if it exists."""
        if not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text())
            for nd in data.get("nodes", []):
                node = Node.from_dict(nd)
                self.nodes[node.id] = node
            for ed in data.get("edges", []):
                edge = Edge.from_dict(ed)
                self.edges[edge.key] = edge
            log.info(f"Graph loaded: {len(self.nodes)} nodes, {len(self.edges)} edges")
        except Exception as e:
            log.error(f"Failed to load graph: {e}")

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self.nodes.clear()
        self.edges.clear()


def build_folder_tree(kb_dir: Path, canon_dir: Path) -> list[tuple[Node, Edge]]:
    """Build folder hierarchy nodes and PARENT_CHILD edges.

    Scans kb_dir and canon_dir, creates a Node for each folder,
    and edges for parent→child relationships. Also checks for README.md
    or index.md in each folder to use as folder summary.
    Returns list of (folder_node, parent_edge) pairs for adding to a graph.
    """
    folder_nodes: dict[str, Node] = {}
    folder_edges: list[Edge] = []

    for base_dir, source in [(kb_dir, "knowledge"), (canon_dir, "canon")]:
        if not base_dir.exists():
            continue
        # Collect all directories that contain .md files
        md_dirs: set[Path] = set()
        for md_file in base_dir.rglob("**/*.md"):
            md_dirs.add(md_file.parent)
            # Also include all parent directories up to base_dir
            d = md_file.parent
            while d != base_dir and d != Path("/"):
                md_dirs.add(d)
                d = d.parent

        for folder_path in md_dirs:
            rel = str(folder_path.relative_to(base_dir))
            folder_id = f"folder_{source}_{rel.replace('/', '_')}"

            # Check for README.md or index.md in this folder for summary
            folder_summary = ""
            for readme_name in ["README.md", "index.md", "_index.md"]:
                readme_path = folder_path / readme_name
                if readme_path.exists():
                    try:
                        content = readme_path.read_text(encoding="utf-8")
                        # Take first meaningful line as summary
                        for line in content.split("\n"):
                            stripped = line.strip()
                            if stripped and not stripped.startswith("#") and not stripped.startswith("---"):
                                folder_summary = stripped[:300]
                                break
                    except (OSError, UnicodeDecodeError):
                        pass
                    break

            # Collect file names in this folder (not subfolders)
            file_names = [f.name for f in folder_path.iterdir()
                         if f.is_file() and f.suffix == ".md"]

            # B1 medallion tier — derived from source + path:
            # canon/*           → canon  (gold)
            # knowledge/wiki/*  → wiki   (silver)
            # knowledge/memory/* → memory (between raw and silver — agent-
            #                     synthesized notes, distilled from threads
            #                     but not yet promoted to wiki)
            # knowledge/raw/*   → raw    (bronze)
            # other knowledge/* → wiki   (legacy fallback)
            # Kept inline (not imported from index.py) to avoid a graph→index
            # dependency cycle.
            rel_norm = rel.replace("\\", "/").lstrip("/")
            if source == "canon":
                folder_tier = "canon"
            elif rel_norm == "raw" or rel_norm.startswith("raw/"):
                folder_tier = "raw"
            elif rel_norm == "memory" or rel_norm.startswith("memory/"):
                folder_tier = "memory"
            else:
                folder_tier = "wiki"

            folder_nodes[folder_id] = Node(
                id=folder_id,
                node_type=NodeType.FOLDER,
                name=rel,
                summary=folder_summary,
                attributes={
                    "source": source,
                    "path": str(folder_path),
                    "file_count": len(file_names),
                    "files": file_names[:20],  # Cap at 20 for metadata size
                    "tier": folder_tier,
                },
                tags=[source, f"tier:{folder_tier}"]
                    + ([rel.split("/")[0]] if "/" in rel else []),
            )

        # Build parent→child edges
        for fid, fnode in list(folder_nodes.items()):
            if fnode.attributes.get("source") != source:
                continue
            # Find parent folder from the node name (which is the relative path)
            name = fnode.name
            if "/" not in name:
                continue  # Top-level folder, no parent
            parent_name = "/".join(name.split("/")[:-1])
            parent_id = f"folder_{source}_{parent_name.replace('/', '_')}"
            if parent_id in folder_nodes:
                folder_edges.append(Edge(
                    source_id=parent_id,
                    target_id=fid,
                    edge_type=EdgeType.PARENT_CHILD,
                    weight=1.0,
                    evidence="folder hierarchy",
                ))

    return [(folder_nodes[fid], None) for fid in folder_nodes] + \
           [(None, edge) for edge in folder_edges]


def format_folder_tree(
    graph: "KnowledgeGraph",
    source: str = "knowledge",
    root_path: str | None = None,
) -> str:
    """Format the folder hierarchy as a tree for LLM consumption.

    Returns indented tree string showing folder structure with file counts.

    Args:
        graph: KnowledgeGraph instance.
        source: ``"canon"`` or ``"knowledge"``.
        root_path: Optional sub-path inside ``source`` to re-root the tree
            (e.g. ``"mind-en-place"``). When provided, only that folder
            and its descendants render. Returns ``""`` when no folder
            matches.
    """
    folders = {nid: n for nid, n in graph.nodes.items()
               if n.node_type == NodeType.FOLDER and n.attributes.get("source") == source}

    if not folders:
        return ""

    # Count files per folder
    file_counts: dict[str, int] = {}
    for node in graph.nodes.values():
        if node.node_type == NodeType.CHUNK:
            folder_tag = node.attributes.get("folder", "")
            if folder_tag:
                file_counts[folder_tag] = file_counts.get(folder_tag, 0) + 1

    # Build tree from folder nodes and their hierarchy edges
    parent_map: dict[str, list[str]] = {}  # parent_id -> [child_ids]
    root_ids = set(folders.keys())
    for edge in graph.edges.values():
        if edge.edge_type == EdgeType.PARENT_CHILD and edge.source_id in folders and edge.target_id in folders:
            parent_map.setdefault(edge.source_id, []).append(edge.target_id)
            root_ids.discard(edge.target_id)

    # When a root_path is requested, narrow the rendering to the folder
    # whose ``name`` matches that path. Falls back to "" so the tools
    # layer can produce a clean error.
    if root_path:
        rp = root_path.replace("\\", "/").strip("/")
        matched_id = next(
            (fid for fid, fnode in folders.items() if fnode.name == rp),
            None,
        )
        if matched_id is None:
            return ""
        # Render from the matched folder only — header line names the path.
        if source == "canon":
            lines = [
                f"[canon] {source}/{rp}/  "
                f"(gold tier — subtree of canon)"
            ]
        else:
            lines = [
                f"[knowledge] {source}/{rp}/  "
                f"(subtree of knowledge)"
            ]
        seen: set = set()

        def _render(folder_id: str, indent: str):
            if folder_id in seen:
                return
            seen.add(folder_id)
            folder = folders.get(folder_id)
            if not folder:
                return
            name = folder.name
            count = file_counts.get(name, 0)
            summary = (folder.summary or "").strip()
            suffix = f" — {summary}" if summary else ""
            tier = folder.attributes.get("tier", "")
            tier_badge = f" [{tier}]" if tier else ""
            lines.append(
                f"{indent}├── {name.split('/')[-1]}/{tier_badge} ({count} files){suffix}"
            )
            for child_id in parent_map.get(folder_id, []):
                _render(child_id, indent + "│   ")

        _render(matched_id, "")
        return "\n".join(lines)

    # Source-level tier label so the agent sees what it's looking at:
    # canon    → gold tier root.
    # knowledge → silver (wiki/), memory (memory/), and bronze (raw/) live here.
    if source == "canon":
        lines = [f"[canon] {source}/  (gold tier — read-only canonical content)"]
    else:
        lines = [
            f"[knowledge] {source}/  (silver tier under wiki/, "
            f"memory tier under memory/, bronze tier under raw/)"
        ]
    seen = set()

    def _render(folder_id: str, indent: str):
        if folder_id in seen:
            return
        seen.add(folder_id)
        folder = folders.get(folder_id)
        if not folder:
            return
        name = folder.name
        count = file_counts.get(name, 0)
        summary = (folder.summary or "").strip()
        suffix = f" — {summary}" if summary else ""
        # P0.5: surface medallion tier badge per folder so the agent can
        # tell wiki/ from raw/ at a glance.
        tier = folder.attributes.get("tier", "")
        tier_badge = f" [{tier}]" if tier else ""
        lines.append(
            f"{indent}├── {name.split('/')[-1]}/{tier_badge} ({count} files){suffix}"
        )
        for child_id in parent_map.get(folder_id, []):
            _render(child_id, indent + "│   ")

    # Render from roots
    for rid in sorted(root_ids):
        _render(rid, "")

    return "\n".join(lines)


def extract_entities(
    content: str,
    heading: str,
    model_id: str = "glm-5:cloud",
    model_url: str = "http://host.docker.internal:11434",
) -> list[dict]:
    """Extract entities from a text chunk using langextract.

    Returns list of dicts with keys: class, text, attributes, char_interval.
    Falls back to empty list if langextract fails.
    """
    try:
        import langextract as lx
        from langextract.factory import ModelConfig

        config = ModelConfig(
            model_id=model_id,
            provider="ollama",
            provider_kwargs={"model_url": model_url},
        )

        examples = [
            lx.data.ExampleData(
                text="Machine learning uses neural networks to learn patterns from data.",
                extractions=[
                    lx.data.Extraction(extraction_class="concept", extraction_text="machine learning"),
                    lx.data.Extraction(extraction_class="technology", extraction_text="neural networks"),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="uses",
                        attributes={"subject": "machine learning", "object": "neural networks", "context": "learning patterns from data"},
                    ),
                ],
            ),
            lx.data.ExampleData(
                text="The hippocampus is a brain region critical for memory formation and spatial navigation.",
                extractions=[
                    lx.data.Extraction(extraction_class="concept", extraction_text="memory formation"),
                    lx.data.Extraction(extraction_class="anatomy", extraction_text="hippocampus"),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="critical for",
                        attributes={"subject": "hippocampus", "object": "memory formation", "context": "brain region supporting memory"},
                    ),
                ],
            ),
        ]

        result = lx.extract(
            text_or_documents=content[:3000],
            prompt_description=(
                "Extract key concepts, entities, technologies, methods, AND relationships between them "
                "from this knowledge base section. For relationships, use extraction_class='relationship' "
                "with attributes containing 'subject' and 'object' (entity names) and 'context' (how they relate). "
                "Focus on domain-specific terms and their connections."
            ),
            examples=examples,
            config=config,
            fence_output=False,
            use_schema_constraints=False,
            extraction_passes=1,
            max_workers=1,
        )

        entities = []
        for e in result.extractions:
            if e.char_interval is None and e.extraction_class != "relationship":
                continue  # Skip ungrounded extractions (but allow relationships even without char_interval)
            entities.append({
                "class": e.extraction_class,
                "text": e.extraction_text,
                "attributes": e.attributes or {},
            })

        return entities

    except Exception as e:
        log.warning(f"Entity extraction failed for '{heading}': {e}")
        return []
"""
Graph Builder — Constructs a knowledge graph with Leiden community detection
and pre-generated community summaries (GraphRAG approach).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import litellm
import networkx as nx
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from ingestion.document_processor import DocumentChunk
from ingestion.entity_extractor import Entity, Relationship

logger = logging.getLogger(__name__)

COMMUNITY_PROMPT = (
    Path(__file__).parent.parent / "config" / "prompts" / "community_summary.txt"
).read_text()


@dataclass
class CommunityInfo:
    community_id: int
    level: int
    entity_names: list[str]
    summary: str = ""
    size: int = 0


@dataclass
class KnowledgeGraph:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    communities: dict[int, CommunityInfo] = field(default_factory=dict)
    entity_to_community: dict[str, int] = field(default_factory=dict)
    chunk_to_entities: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_node(self, name: str) -> Optional[dict]:
        name_lower = name.lower()
        for node in self.graph.nodes:
            if node.lower() == name_lower:
                return dict(self.graph.nodes[node])
        return None

    def get_neighbors(self, name: str, max_hops: int = 2) -> list[dict]:
        """BFS traversal up to max_hops, returning connected nodes with edge info."""
        name_lower = name.lower()
        start = None
        for node in self.graph.nodes:
            if node.lower() == name_lower:
                start = node
                break
        if start is None:
            return []

        visited = set()
        result = []
        queue = [(start, 0)]

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_hops:
                continue
            visited.add(current)

            if current != start:
                result.append(
                    {
                        "node": current,
                        "depth": depth,
                        **dict(self.graph.nodes[current]),
                    }
                )

            if depth < max_hops:
                # Outgoing edges
                for _, target, data in self.graph.out_edges(current, data=True):
                    if target not in visited:
                        conf = data.get("confidence", 0.5)
                        if conf >= settings.graph_confidence_threshold:
                            queue.append((target, depth + 1))

                # Incoming edges (bidirectional traversal)
                for source, _, data in self.graph.in_edges(current, data=True):
                    if source not in visited:
                        conf = data.get("confidence", 0.5)
                        if conf >= settings.graph_confidence_threshold:
                            queue.append((source, depth + 1))

        return result

    def get_community_summary(self, entity_name: str) -> Optional[str]:
        """Get the community summary for an entity's community."""
        name_lower = entity_name.lower()
        for ent, comm_id in self.entity_to_community.items():
            if ent.lower() == name_lower:
                comm = self.communities.get(comm_id)
                return comm.summary if comm else None
        return None

    def to_dict(self) -> dict:
        """Serialize the full KG to a dict for JSON persistence."""
        nodes = []
        for node, data in self.graph.nodes(data=True):
            nodes.append({"name": node, **data})

        edges = []
        for src, tgt, data in self.graph.edges(data=True):
            edges.append({"source": src, "target": tgt, **data})

        communities = {
            str(k): {
                "community_id": v.community_id,
                "level": v.level,
                "entity_names": v.entity_names,
                "summary": v.summary,
                "size": v.size,
            }
            for k, v in self.communities.items()
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "communities": communities,
            "entity_to_community": self.entity_to_community,
            "chunk_to_entities": self.chunk_to_entities,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeGraph:
        """Deserialize from dict."""
        kg = cls()
        for node in data.get("nodes", []):
            name = node.pop("name")
            kg.graph.add_node(name, **node)

        for edge in data.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            kg.graph.add_edge(src, tgt, **edge)

        for k, v in data.get("communities", {}).items():
            kg.communities[int(k)] = CommunityInfo(**v)

        kg.entity_to_community = data.get("entity_to_community", {})
        kg.chunk_to_entities = data.get("chunk_to_entities", {})
        kg.metadata = data.get("metadata", {})
        return kg


class GraphBuilder:
    """Builds a knowledge graph from entities/relationships with community detection."""

    def __init__(self, model: str = settings.fast_model):
        self.model = model

    def build(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        chunks: list[DocumentChunk],
    ) -> KnowledgeGraph:
        """Build the full knowledge graph with communities."""
        kg = KnowledgeGraph()
        kg.metadata = {
            "built_at": datetime.utcnow().isoformat(),
            "num_entities": len(entities),
            "num_relationships": len(relationships),
            "num_chunks": len(chunks),
        }

        # Add nodes
        for ent in entities:
            kg.graph.add_node(
                ent.name,
                entity_type=ent.entity_type,
                description=ent.description,
                source_chunks=ent.source_chunks,
            )
            # Track chunk -> entity mapping
            for cid in ent.source_chunks:
                kg.chunk_to_entities.setdefault(cid, []).append(ent.name)

        # Add edges with metadata
        for rel in relationships:
            if kg.graph.has_node(rel.source) and kg.graph.has_node(rel.target):
                kg.graph.add_edge(
                    rel.source,
                    rel.target,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    confidence=rel.confidence,
                    source_chunks=rel.source_chunks,
                )

        # Detect communities using Leiden algorithm
        self._detect_communities(kg)

        logger.info(
            f"Built KG: {kg.graph.number_of_nodes()} nodes, "
            f"{kg.graph.number_of_edges()} edges, "
            f"{len(kg.communities)} communities"
        )
        return kg

    def _detect_communities(self, kg: KnowledgeGraph) -> None:
        """Run Leiden community detection on the graph."""
        if kg.graph.number_of_nodes() < 2:
            return

        try:
            # Use graspologic's Leiden implementation
            from graspologic.partition import leiden

            undirected = kg.graph.to_undirected()
            partition = leiden(undirected, resolution=settings.leiden_resolution)

            # Build community structures
            community_members: dict[int, list[str]] = {}
            for node, comm_id in partition.items():
                community_members.setdefault(comm_id, []).append(node)
                kg.entity_to_community[node] = comm_id

            for comm_id, members in community_members.items():
                kg.communities[comm_id] = CommunityInfo(
                    community_id=comm_id,
                    level=0,
                    entity_names=members,
                    size=len(members),
                )

        except ImportError:
            logger.warning("graspologic not available; falling back to NetworkX Louvain")
            try:
                from networkx.algorithms.community import louvain_communities

                undirected = kg.graph.to_undirected()
                communities = louvain_communities(
                    undirected,
                    resolution=settings.leiden_resolution,
                )

                for comm_id, members in enumerate(communities):
                    member_list = list(members)
                    for node in member_list:
                        kg.entity_to_community[node] = comm_id
                    kg.communities[comm_id] = CommunityInfo(
                        community_id=comm_id,
                        level=0,
                        entity_names=member_list,
                        size=len(member_list),
                    )
            except Exception as e:
                logger.error(f"Community detection failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def _generate_community_summary(
        self,
        community: CommunityInfo,
        kg: KnowledgeGraph,
        chunks: list[DocumentChunk],
    ) -> str:
        """Generate a summary for a community using LLM."""
        # Gather entity descriptions
        entity_info = []
        for name in community.entity_names[:20]:  # Limit for context window
            node_data = kg.graph.nodes.get(name, {})
            entity_info.append(
                f"- {name} ({node_data.get('entity_type', 'unknown')}): "
                f"{node_data.get('description', 'no description')}"
            )

        # Gather relationship descriptions within community
        rel_info = []
        members_set = set(community.entity_names)
        for src, tgt, data in kg.graph.edges(data=True):
            if src in members_set and tgt in members_set:
                rel_info.append(
                    f"- {src} --[{data.get('relation_type', 'RELATES_TO')}]--> {tgt}: "
                    f"{data.get('description', '')}"
                )

        # Gather representative text chunks
        chunk_map = {c.chunk_id: c.text for c in chunks}
        relevant_chunks = set()
        for name in community.entity_names:
            node_data = kg.graph.nodes.get(name, {})
            for cid in node_data.get("source_chunks", [])[:3]:
                if cid in chunk_map:
                    relevant_chunks.add(chunk_map[cid][:500])

        prompt = (
            COMMUNITY_PROMPT.replace("{entities}", "\n".join(entity_info[:20]))
            .replace("{relationships}", "\n".join(rel_info[:20]))
            .replace("{chunks}", "\n---\n".join(list(relevant_chunks)[:5]))
        )

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    async def generate_all_summaries(
        self,
        kg: KnowledgeGraph,
        chunks: list[DocumentChunk],
    ) -> None:
        """Generate summaries for all communities."""
        import asyncio

        tasks = []
        for comm in kg.communities.values():
            if comm.size >= 2:  # Only summarize non-trivial communities
                tasks.append(self._generate_community_summary(comm, kg, chunks))
            else:
                tasks.append(None)

        results = []
        for task in tasks:
            if task is None:
                results.append(None)
            else:
                results.append(await task)

        for comm, summary in zip(kg.communities.values(), results):
            if summary:
                comm.summary = summary

        logger.info(f"Generated summaries for {sum(1 for r in results if r)} communities")

    def save(self, kg: KnowledgeGraph, path: Optional[Path] = None) -> None:
        """Persist the knowledge graph to JSON."""
        path = path or settings.graph_persist_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(kg.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved KG to {path}")

    def load(self, path: Optional[Path] = None) -> KnowledgeGraph:
        """Load knowledge graph from JSON."""
        path = path or settings.graph_persist_path
        with open(path) as f:
            data = json.load(f)
        kg = KnowledgeGraph.from_dict(data)
        logger.info(
            f"Loaded KG: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges"
        )
        return kg

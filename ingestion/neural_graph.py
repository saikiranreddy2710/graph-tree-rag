"""
Neural Graph — Unified vector-graph structure inspired by the human nervous system.

Every node is both a vector embedding and a graph node. Edges carry typed,
weighted relationships. Retrieval uses spreading activation rather than
separate vector search + graph walk + fusion.

Research basis:
  - Graph Neural Networks (message passing between embedded nodes)
  - Knowledge Graph Embeddings (TransE / RotatE — entities & relations in same space)
  - Spreading Activation (cognitive science model of human memory retrieval)
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import faiss
import networkx as nx
import numpy as np

from config.settings import settings
from ingestion.document_processor import DocumentChunk
from ingestion.entity_extractor import Entity, Relationship

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────


@dataclass
class NeuralNode:
    """A neuron in the neural graph — holds content, embedding, and activation state."""

    node_id: str
    text: str
    embedding: Optional[np.ndarray] = None  # 384-dim vector
    node_type: str = "chunk"  # chunk | entity | community_summary | tree_summary
    entity_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    activation: float = 0.0  # transient — used during retrieval

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "node_type": self.node_type,
            "entity_names": self.entity_names,
            "metadata": self.metadata,
            # embedding stored separately in FAISS index
        }

    @classmethod
    def from_dict(cls, data: dict) -> NeuralNode:
        return cls(
            node_id=data["node_id"],
            text=data["text"],
            node_type=data.get("node_type", "chunk"),
            entity_names=data.get("entity_names", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class NeuralEdge:
    """A synapse in the neural graph — carries typed, weighted connections."""

    source_id: str
    target_id: str
    edge_type: str  # entity_link | semantic_similarity | hierarchical | co_occurrence
    weight: float = 0.5  # propagation strength (0.0 to 1.0)
    relationship_desc: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "relationship_desc": self.relationship_desc,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NeuralEdge:
        return cls(**data)


# ── Neural Graph ──────────────────────────────────────────────────────────


class NeuralGraph:
    """
    Unified vector-graph structure. Like the brain's nervous system:
    - Nodes (neurons) hold embeddings + content
    - Edges (synapses) carry typed, weighted connections
    - Retrieval = spreading activation through the graph
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes: dict[str, NeuralNode] = {}
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self._faiss_id_to_node_id: dict[int, str] = {}
        self._node_id_to_faiss_id: dict[str, int] = {}
        self.metadata: dict[str, Any] = {}

    def add_node(self, node: NeuralNode) -> None:
        """Add a neural node to the graph."""
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            node_type=node.node_type,
            entity_names=node.entity_names,
        )

    def add_edge(self, edge: NeuralEdge) -> None:
        """Add a neural edge (synapse) to the graph."""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            return
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type,
            weight=edge.weight,
            relationship_desc=edge.relationship_desc,
            confidence=edge.confidence,
        )

    def build_faiss_index(self) -> None:
        """Build a FAISS index over all node embeddings for fast initial activation."""
        embedded_nodes = [
            (nid, node) for nid, node in self.nodes.items() if node.embedding is not None
        ]
        if not embedded_nodes:
            logger.warning("No embedded nodes to index")
            return

        dim = embedded_nodes[0][1].embedding.shape[0]
        embeddings = np.array(
            [node.embedding for _, node in embedded_nodes], dtype=np.float32
        )

        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)

        self._faiss_id_to_node_id = {i: nid for i, (nid, _) in enumerate(embedded_nodes)}
        self._node_id_to_faiss_id = {nid: i for i, (nid, _) in enumerate(embedded_nodes)}

        logger.info(f"Neural graph FAISS index: {self.faiss_index.ntotal} vectors")

    def get_neighbors(self, node_id: str) -> list[tuple[str, dict]]:
        """Get all neighbors of a node with edge data."""
        neighbors = []
        # Outgoing
        for _, target, data in self.graph.out_edges(node_id, data=True):
            neighbors.append((target, dict(data)))
        # Incoming
        for source, _, data in self.graph.in_edges(node_id, data=True):
            neighbors.append((source, dict(data)))
        return neighbors

    def find_nearest_nodes(
        self, query_embedding: np.ndarray, top_k: int = 15
    ) -> list[tuple[str, float]]:
        """FAISS search to find initial seed nodes for activation."""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.faiss_index.search(
            query_embedding.astype(np.float32), min(top_k, self.faiss_index.ntotal)
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            node_id = self._faiss_id_to_node_id.get(int(idx))
            if node_id:
                results.append((node_id, float(score)))
        return results

    def reset_activations(self) -> None:
        """Reset all node activations to zero (called before each retrieval)."""
        for node in self.nodes.values():
            node.activation = 0.0

    def stats(self) -> dict:
        """Return graph statistics."""
        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        node_types = {}
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "faiss_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
        }

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize graph structure (embeddings stored separately)."""
        nodes = [node.to_dict() for node in self.nodes.values()]
        edges = []
        for src, tgt, data in self.graph.edges(data=True):
            edges.append(
                NeuralEdge(
                    source_id=src,
                    target_id=tgt,
                    edge_type=data.get("edge_type", "unknown"),
                    weight=data.get("weight", 0.5),
                    relationship_desc=data.get("relationship_desc", ""),
                    confidence=data.get("confidence", 1.0),
                ).to_dict()
            )
        return {"nodes": nodes, "edges": edges, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict) -> NeuralGraph:
        """Deserialize from dict."""
        ng = cls()
        ng.metadata = data.get("metadata", {})
        for node_data in data.get("nodes", []):
            ng.add_node(NeuralNode.from_dict(node_data))
        for edge_data in data.get("edges", []):
            edge = NeuralEdge.from_dict(edge_data)
            ng.add_edge(edge)
        return ng

    def save(self, path: Optional[Path] = None) -> None:
        """Persist neural graph to disk."""
        path = path or settings.neural_graph_persist_path
        path.mkdir(parents=True, exist_ok=True)

        # Save structure
        with open(path / "graph.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        # Save embeddings
        embeddings_data = {}
        for nid, node in self.nodes.items():
            if node.embedding is not None:
                embeddings_data[nid] = node.embedding
        with open(path / "embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_data, f)

        # Save FAISS index
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(path / "neural.faiss"))
            with open(path / "faiss_mapping.json", "w") as f:
                json.dump(
                    {
                        "id_to_node": {str(k): v for k, v in self._faiss_id_to_node_id.items()},
                        "node_to_id": {k: v for k, v in self._node_id_to_faiss_id.items()},
                    },
                    f,
                )

        logger.info(f"Saved neural graph to {path}: {self.stats()}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> NeuralGraph:
        """Load neural graph from disk."""
        path = path or settings.neural_graph_persist_path

        with open(path / "graph.json") as f:
            data = json.load(f)
        ng = cls.from_dict(data)

        # Load embeddings
        emb_path = path / "embeddings.pkl"
        if emb_path.exists():
            with open(emb_path, "rb") as f:
                embeddings_data = pickle.load(f)
            for nid, emb in embeddings_data.items():
                if nid in ng.nodes:
                    ng.nodes[nid].embedding = emb

        # Load FAISS index
        faiss_path = path / "neural.faiss"
        if faiss_path.exists():
            ng.faiss_index = faiss.read_index(str(faiss_path))
            mapping_path = path / "faiss_mapping.json"
            if mapping_path.exists():
                with open(mapping_path) as f:
                    mapping = json.load(f)
                ng._faiss_id_to_node_id = {int(k): v for k, v in mapping["id_to_node"].items()}
                ng._node_id_to_faiss_id = mapping["node_to_id"]

        logger.info(f"Loaded neural graph from {path}: {ng.stats()}")
        return ng


# ── Neural Graph Builder ─────────────────────────────────────────────────


class NeuralGraphBuilder:
    """
    Builds a NeuralGraph from chunks, entities, relationships, and tree nodes.
    Creates 4 edge types:
      1. entity_link      — same entity in multiple chunks
      2. semantic_similarity — high cosine similarity between chunk embeddings
      3. hierarchical     — RAPTOR tree parent→child connections
      4. co_occurrence    — entities co-occurring in the same chunk
    """

    def __init__(
        self,
        similarity_threshold: float = settings.neural_graph_similarity_threshold,
    ):
        self.similarity_threshold = similarity_threshold
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(settings.embedding_model)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        embedder = self._get_embedder()
        return embedder.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        ).astype(np.float32)

    def build(
        self,
        chunks: list[DocumentChunk],
        entities: list[Entity],
        relationships: list[Relationship],
        tree=None,  # Optional[RaptorTree]
    ) -> NeuralGraph:
        """Build the full neural graph from all data sources."""
        ng = NeuralGraph()
        ng.metadata = {
            "built_at": datetime.utcnow().isoformat(),
            "num_chunks": len(chunks),
            "num_entities": len(entities),
            "num_relationships": len(relationships),
        }

        logger.info("Building neural graph...")

        # Step 1: Create chunk nodes with embeddings
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = self._embed(chunk_texts) if chunk_texts else np.array([])

        for i, chunk in enumerate(chunks):
            # Determine which entities are in this chunk
            chunk_entities = []
            for ent in entities:
                if chunk.chunk_id in ent.source_chunks:
                    chunk_entities.append(ent.name)

            node = NeuralNode(
                node_id=chunk.chunk_id,
                text=chunk.text,
                embedding=chunk_embeddings[i] if len(chunk_embeddings) > 0 else None,
                node_type="chunk",
                entity_names=chunk_entities,
                metadata={
                    "source": chunk.metadata.source,
                    "section": chunk.metadata.section_header,
                },
            )
            ng.add_node(node)

        # Step 2: Create entity nodes
        entity_texts = [f"{e.name}: {e.description}" for e in entities]
        entity_embeddings = self._embed(entity_texts) if entity_texts else np.array([])

        for i, ent in enumerate(entities):
            node = NeuralNode(
                node_id=f"entity_{ent.name.lower().replace(' ', '_')}",
                text=f"{ent.name} ({ent.entity_type}): {ent.description}",
                embedding=entity_embeddings[i] if len(entity_embeddings) > 0 else None,
                node_type="entity",
                entity_names=[ent.name],
                metadata={"entity_type": ent.entity_type},
            )
            ng.add_node(node)

        # Step 3: Add tree summary nodes (if RAPTOR tree provided)
        if tree:
            self._add_tree_nodes(ng, tree)

        # Step 4: Create edges
        self._create_entity_link_edges(ng, entities, chunks)
        self._create_relationship_edges(ng, relationships)
        self._create_semantic_similarity_edges(ng, chunks, chunk_embeddings)
        self._create_co_occurrence_edges(ng, entities, chunks)
        if tree:
            self._create_hierarchical_edges(ng, tree)

        # Step 5: Build FAISS index
        ng.build_faiss_index()

        logger.info(f"Neural graph built: {ng.stats()}")
        return ng

    def _add_tree_nodes(self, ng: NeuralGraph, tree) -> None:
        """Add RAPTOR tree summary nodes."""
        for node_id, tree_node in tree.nodes.items():
            if tree_node.level > 0:  # Skip leaves (already as chunk nodes)
                emb = (
                    np.array(tree_node.embedding, dtype=np.float32)
                    if tree_node.embedding
                    else self._embed([tree_node.text])[0]
                )
                node = NeuralNode(
                    node_id=node_id,
                    text=tree_node.text,
                    embedding=emb,
                    node_type="tree_summary",
                    metadata={"level": tree_node.level},
                )
                ng.add_node(node)

    def _create_entity_link_edges(
        self, ng: NeuralGraph, entities: list[Entity], chunks: list[DocumentChunk]
    ) -> None:
        """Edge type 1: Same entity appears in multiple chunks → connect those chunks."""
        for ent in entities:
            chunk_ids = [cid for cid in ent.source_chunks if cid in ng.nodes]
            entity_node_id = f"entity_{ent.name.lower().replace(' ', '_')}"

            # Connect entity node to each chunk it appears in
            for cid in chunk_ids:
                if entity_node_id in ng.nodes:
                    ng.add_edge(NeuralEdge(
                        source_id=entity_node_id,
                        target_id=cid,
                        edge_type="entity_link",
                        weight=0.8,
                        relationship_desc=f"Entity '{ent.name}' appears in chunk",
                        confidence=1.0,
                    ))

            # Connect chunks that share this entity
            for i, cid_a in enumerate(chunk_ids):
                for cid_b in chunk_ids[i + 1:]:
                    # Bidirectional with moderate weight
                    ng.add_edge(NeuralEdge(
                        source_id=cid_a,
                        target_id=cid_b,
                        edge_type="entity_link",
                        weight=0.6,
                        relationship_desc=f"Shared entity: {ent.name}",
                    ))
                    ng.add_edge(NeuralEdge(
                        source_id=cid_b,
                        target_id=cid_a,
                        edge_type="entity_link",
                        weight=0.6,
                        relationship_desc=f"Shared entity: {ent.name}",
                    ))

    def _create_relationship_edges(
        self, ng: NeuralGraph, relationships: list[Relationship]
    ) -> None:
        """Create edges from extracted entity-entity relationships."""
        for rel in relationships:
            src_id = f"entity_{rel.source.lower().replace(' ', '_')}"
            tgt_id = f"entity_{rel.target.lower().replace(' ', '_')}"
            if src_id in ng.nodes and tgt_id in ng.nodes:
                ng.add_edge(NeuralEdge(
                    source_id=src_id,
                    target_id=tgt_id,
                    edge_type="entity_link",
                    weight=rel.confidence * 0.9,
                    relationship_desc=f"{rel.source} --[{rel.relation_type}]--> {rel.target}: {rel.description}",
                    confidence=rel.confidence,
                ))

    def _create_semantic_similarity_edges(
        self,
        ng: NeuralGraph,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Edge type 2: Chunks with high cosine similarity get connected."""
        if len(embeddings) < 2:
            return

        # Compute pairwise similarities efficiently
        sim_matrix = embeddings @ embeddings.T

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                sim = float(sim_matrix[i, j])
                if sim >= self.similarity_threshold:
                    weight = (sim - self.similarity_threshold) / (1.0 - self.similarity_threshold)
                    weight = min(weight * 0.7, 0.7)  # Cap at 0.7

                    ng.add_edge(NeuralEdge(
                        source_id=chunks[i].chunk_id,
                        target_id=chunks[j].chunk_id,
                        edge_type="semantic_similarity",
                        weight=weight,
                        relationship_desc=f"Semantic similarity: {sim:.3f}",
                        confidence=sim,
                    ))
                    ng.add_edge(NeuralEdge(
                        source_id=chunks[j].chunk_id,
                        target_id=chunks[i].chunk_id,
                        edge_type="semantic_similarity",
                        weight=weight,
                        relationship_desc=f"Semantic similarity: {sim:.3f}",
                        confidence=sim,
                    ))

    def _create_co_occurrence_edges(
        self, ng: NeuralGraph, entities: list[Entity], chunks: list[DocumentChunk]
    ) -> None:
        """Edge type 4: Entities co-occurring in the same chunk → entity-entity edge."""
        # Build chunk → entity mapping
        chunk_entities: dict[str, list[str]] = {}
        for ent in entities:
            for cid in ent.source_chunks:
                chunk_entities.setdefault(cid, []).append(ent.name)

        # Create edges between co-occurring entities
        for cid, ent_names in chunk_entities.items():
            for i, name_a in enumerate(ent_names):
                for name_b in ent_names[i + 1:]:
                    src_id = f"entity_{name_a.lower().replace(' ', '_')}"
                    tgt_id = f"entity_{name_b.lower().replace(' ', '_')}"
                    if src_id in ng.nodes and tgt_id in ng.nodes:
                        ng.add_edge(NeuralEdge(
                            source_id=src_id,
                            target_id=tgt_id,
                            edge_type="co_occurrence",
                            weight=0.5,
                            relationship_desc=f"Co-occur in chunk {cid}",
                        ))
                        ng.add_edge(NeuralEdge(
                            source_id=tgt_id,
                            target_id=src_id,
                            edge_type="co_occurrence",
                            weight=0.5,
                            relationship_desc=f"Co-occur in chunk {cid}",
                        ))

    def _create_hierarchical_edges(self, ng: NeuralGraph, tree) -> None:
        """Edge type 3: RAPTOR tree parent→child connections."""
        for node_id, tree_node in tree.nodes.items():
            if node_id not in ng.nodes:
                continue
            for child_id in tree_node.children:
                child_node_id = child_id
                # Map leaf children to chunk IDs if needed
                if child_id not in ng.nodes:
                    child_tree_node = tree.nodes.get(child_id)
                    if child_tree_node and child_tree_node.source_chunk_id:
                        child_node_id = child_tree_node.source_chunk_id
                if child_node_id in ng.nodes:
                    ng.add_edge(NeuralEdge(
                        source_id=node_id,
                        target_id=child_node_id,
                        edge_type="hierarchical",
                        weight=0.75,
                        relationship_desc="RAPTOR tree parent→child",
                    ))
                    ng.add_edge(NeuralEdge(
                        source_id=child_node_id,
                        target_id=node_id,
                        edge_type="hierarchical",
                        weight=0.5,  # Weaker child→parent
                        relationship_desc="RAPTOR tree child→parent",
                    ))

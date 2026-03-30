"""
CortexStore — Brain-inspired storage model for Graph-Tree RAG.

Unlike traditional vector stores or knowledge graphs, CortexStore mimics
the brain's nervous system behavior:

  - CortexNodes (neurons): hold embeddings + content + activation history
  - Synapses (connections): learn from usage via Hebbian plasticity
  - CorticalColumns: self-organizing clusters of related knowledge
  - WorkingMemory: capacity-limited activation buffer per query session
  - Consolidation: merge similar nodes, prune weak synapses, strengthen paths

The structure evolves with every query — connections that are used together
strengthen, unused ones decay, and similar knowledge consolidates over time.
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

logger = logging.getLogger(__name__)


# ── CortexNode — A Neuron ────────────────────────────────────────────


@dataclass
class CortexNode:
    """
    A neuron in the cortex. Holds content, embedding, and activation history.
    Unlike static vector entries, nodes track their own usage patterns.
    """

    node_id: str
    text: str
    embedding: Optional[np.ndarray] = None
    node_type: str = "chunk"  # chunk | entity | summary | consolidated
    entity_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # TurboQuant Compressed state
    polar_q: Optional[np.ndarray] = None
    qjl_signs: Optional[np.ndarray] = None

    # Brain-like properties
    activation: float = 0.0              # transient — current query activation
    access_count: int = 0                # how many times this node was retrieved
    last_accessed: Optional[str] = None  # ISO timestamp of last retrieval
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def fire(self, activation_value: float) -> None:
        """Activate this neuron (like a neuron firing)."""
        self.activation = activation_value
        self.access_count += 1
        self.last_accessed = datetime.utcnow().isoformat()

    def get_embedding(self, compressor: Optional[Any] = None) -> Optional[np.ndarray]:
        """Get the float32 embedding, decompressing on-the-fly if needed to save RAM."""
        if self.embedding is not None:
            return self.embedding
        if self.polar_q is not None and self.qjl_signs is not None and compressor is not None:
            # Expand to 2D for compressor, extract 1D back
            return compressor.decompress(self.polar_q.reshape(1, -1), self.qjl_signs.reshape(1, -1))[0]
        return None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "node_type": self.node_type,
            "entity_names": self.entity_names,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CortexNode:
        return cls(
            node_id=data["node_id"],
            text=data["text"],
            node_type=data.get("node_type", "chunk"),
            entity_names=data.get("entity_names", []),
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )


# ── Synapse — A Connection That Learns ───────────────────────────────


@dataclass
class Synapse:
    """
    A connection between two neurons that learns from usage.

    Implements Hebbian plasticity: "neurons that fire together wire together."
    Weight strengthens when both connected nodes are co-retrieved,
    and decays during consolidation if unused.
    """

    source_id: str
    target_id: str
    synapse_type: str  # entity_link | semantic | hierarchical | co_occurrence
    weight: float = 0.5
    initial_weight: float = 0.5
    confidence: float = 1.0
    relationship_desc: str = ""

    # Learning properties
    co_fire_count: int = 0      # times both endpoints activated together
    last_fired: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def hebbian_update(
        self, src_activation: float, tgt_activation: float, learning_rate: float
    ) -> None:
        """
        Hebbian learning: strengthen connection when both neurons fire.
        ΔW = lr × activation(source) × activation(target)
        """
        delta = learning_rate * src_activation * tgt_activation
        self.weight = min(1.0, self.weight + delta)  # Cap at 1.0
        self.co_fire_count += 1
        self.last_fired = datetime.utcnow().isoformat()

    def decay(self, decay_rate: float) -> None:
        """
        Synaptic decay: unused connections weaken over time.
        Mimics synaptic pruning in the brain.
        """
        self.weight = max(0.0, self.weight * (1.0 - decay_rate))

    @property
    def is_prunable(self) -> bool:
        """Check if this synapse should be pruned (too weak)."""
        return self.weight < settings.cortex_pruning_threshold

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "synapse_type": self.synapse_type,
            "weight": self.weight,
            "initial_weight": self.initial_weight,
            "confidence": self.confidence,
            "relationship_desc": self.relationship_desc,
            "co_fire_count": self.co_fire_count,
            "last_fired": self.last_fired,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Synapse:
        return cls(**data)


# ── CorticalColumn — Self-Organizing Cluster ─────────────────────────


@dataclass
class CorticalColumn:
    """
    A cortical column — a self-organizing cluster of related neurons.

    In the brain, cortical columns are vertical structures that process
    related information. Here, they group semantically related nodes
    and allow fast column-level activation before diving into individual nodes.
    """

    column_id: str
    centroid: Optional[np.ndarray] = None
    member_ids: list[str] = field(default_factory=list)
    summary: str = ""
    activation: float = 0.0  # transient

    def update_centroid(self, node_embeddings: dict[str, np.ndarray]) -> None:
        """Recompute centroid from current member embeddings."""
        member_embs = [
            node_embeddings[nid] for nid in self.member_ids if nid in node_embeddings
        ]
        if member_embs:
            centroid = np.mean(member_embs, axis=0).astype(np.float32)
            self.centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

    def to_dict(self) -> dict:
        return {
            "column_id": self.column_id,
            "member_ids": self.member_ids,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CorticalColumn:
        return cls(
            column_id=data["column_id"],
            member_ids=data.get("member_ids", []),
            summary=data.get("summary", ""),
        )


# ── WorkingMemory — Activation Buffer ────────────────────────────────


class WorkingMemory:
    """
    Working memory — capacity-limited buffer of currently active nodes.

    Like the brain's working memory (7±2 items), this maintains a small
    set of highly relevant nodes from the current query context.
    """

    def __init__(self, capacity: int = settings.cortex_working_memory_capacity):
        self.capacity = capacity
        self.active_nodes: dict[str, float] = {}  # node_id -> activation
        self.context_embedding: Optional[np.ndarray] = None

    def update(self, activated: dict[str, float], embeddings: dict[str, np.ndarray]) -> None:
        """Update working memory with newly activated nodes (keep top-K)."""
        # Merge with existing (new activations take priority)
        merged = {**self.active_nodes}
        for nid, act in activated.items():
            merged[nid] = max(merged.get(nid, 0), act)

        # Keep top-K by activation
        sorted_nodes = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        self.active_nodes = dict(sorted_nodes[: self.capacity])

        # Update context embedding (running average of active node embeddings)
        active_embs = [embeddings[nid] for nid in self.active_nodes if nid in embeddings]
        if active_embs:
            ctx = np.mean(active_embs, axis=0).astype(np.float32)
            self.context_embedding = ctx / (np.linalg.norm(ctx) + 1e-8)

    def reset(self) -> None:
        """Clear working memory (new session)."""
        self.active_nodes = {}
        self.context_embedding = None

    def get_context_boost(self, node_id: str) -> float:
        """Return activation boost for a node if it's in working memory."""
        return self.active_nodes.get(node_id, 0.0) * 0.2  # 20% boost


# ── CortexStore — The Brain ──────────────────────────────────────────


class CortexStore:
    """
    Brain-inspired storage model. Unlike static vector stores:
      - Connections learn from usage (Hebbian plasticity)
      - Knowledge self-organizes into cortical columns
      - Working memory maintains query context
      - Consolidation merges/prunes/strengthens over time
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes: dict[str, CortexNode] = {}
        self.columns: dict[str, CorticalColumn] = {}
        self.working_memory = WorkingMemory()
        
        self.compressor = None
        if settings.cortex_turboquant_enabled:
            from ingestion.turboquant import TurboQuantCompressor
            self.compressor = TurboQuantCompressor(settings.embedding_dimension, settings.cortex_turboquant_bits)

        # FAISS indices
        self._column_index: Optional[faiss.IndexFlatIP] = None
        self._column_id_map: dict[int, str] = {}
        self._node_index: Optional[faiss.IndexFlatIP] = None
        self._node_id_map: dict[int, str] = {}
        self._node_id_reverse: dict[str, int] = {}

        # Consolidation tracking
        self._query_count: int = 0
        self.metadata: dict[str, Any] = {}

    # ── Node / Synapse Management ─────────────────────────────────

    def add_node(self, node: CortexNode) -> None:
        """Add a neuron to the cortex."""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, node_type=node.node_type)

    def add_synapse(self, synapse: Synapse) -> None:
        """Add a synapse (learning connection) between neurons."""
        if synapse.source_id not in self.nodes or synapse.target_id not in self.nodes:
            return
        self.graph.add_edge(
            synapse.source_id,
            synapse.target_id,
            synapse_type=synapse.synapse_type,
            weight=synapse.weight,
            initial_weight=synapse.initial_weight,
            confidence=synapse.confidence,
            relationship_desc=synapse.relationship_desc,
            co_fire_count=synapse.co_fire_count,
            last_fired=synapse.last_fired,
            created_at=synapse.created_at,
        )

    def get_synapse(self, source_id: str, target_id: str) -> Optional[Synapse]:
        """Get a synapse between two nodes."""
        if self.graph.has_edge(source_id, target_id):
            data = dict(self.graph.edges[source_id, target_id])
            return Synapse(source_id=source_id, target_id=target_id, **data)
        return None

    def get_neighbors(self, node_id: str) -> list[tuple[str, dict]]:
        """Get all synaptic connections from a node."""
        neighbors = []
        for _, tgt, data in self.graph.out_edges(node_id, data=True):
            neighbors.append((tgt, dict(data)))
        for src, _, data in self.graph.in_edges(node_id, data=True):
            neighbors.append((src, dict(data)))
        return neighbors

    # ── Cortical Column Construction ──────────────────────────────

    def build_columns(self, n_columns: int = settings.cortex_num_columns) -> None:
        """Build cortical columns via GMM clustering of node embeddings."""
        embedded = [(nid, n.get_embedding(self.compressor)) for nid, n in self.nodes.items() if n.get_embedding(self.compressor) is not None]
        if len(embedded) < n_columns:
            # Too few nodes — one column for all
            col = CorticalColumn(
                column_id="col_0", member_ids=[nid for nid, _ in embedded]
            )
            emb_dict = {nid: emb for nid, emb in embedded}
            col.update_centroid(emb_dict)
            self.columns = {"col_0": col}
            self._build_column_index()
            return

        embs = np.array([emb for _, emb in embedded], dtype=np.float32)
        node_ids = [nid for nid, _ in embedded]

        try:
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(
                n_components=min(n_columns, len(embedded)),
                covariance_type="diag",
                random_state=42,
                max_iter=100,
            )
            labels = gmm.fit_predict(embs)
        except ImportError:
            # Fallback: simple K-means style assignment
            logger.warning("sklearn not available; using random column assignment")
            labels = np.random.randint(0, n_columns, size=len(embedded))

        # Build columns from cluster assignments
        self.columns = {}
        emb_dict = {nid: emb for nid, emb in embedded}

        for col_idx in range(max(labels) + 1):
            members = [node_ids[i] for i, lbl in enumerate(labels) if lbl == col_idx]
            if not members:
                continue
            col = CorticalColumn(
                column_id=f"col_{col_idx}",
                member_ids=members,
            )
            col.update_centroid(emb_dict)
            self.columns[col.column_id] = col

        self._build_column_index()
        logger.info(f"Built {len(self.columns)} cortical columns from {len(embedded)} nodes")

    def _build_column_index(self) -> None:
        """Build FAISS index over column centroids for fast column activation."""
        cols_with_centroids = [
            (cid, col) for cid, col in self.columns.items() if col.centroid is not None
        ]
        if not cols_with_centroids:
            return

        dim = cols_with_centroids[0][1].centroid.shape[0]
        centroids = np.array([col.centroid for _, col in cols_with_centroids], dtype=np.float32)

        self._column_index = faiss.IndexFlatIP(dim)
        self._column_index.add(centroids)
        self._column_id_map = {i: cid for i, (cid, _) in enumerate(cols_with_centroids)}

    def build_node_index(self) -> None:
        """Build FAISS index over all node embeddings."""
        if self.compressor is not None:
            # Native C++ Binary FAISS indexing for extreme speed and memory savings
            embedded = [(nid, n.qjl_signs) for nid, n in self.nodes.items() if n.qjl_signs is not None]
            if not embedded:
                return

            dim_bits = self.compressor.qjl.proj_dim
            self._node_index = faiss.IndexBinaryFlat(dim_bits)
            
            # Add packed uint8 representations directly to C++ memory
            packed_embs = np.array([emb for _, emb in embedded], dtype=np.uint8)
            self._node_index.add(packed_embs)
            self._node_id_map = {i: nid for i, (nid, _) in enumerate(embedded)}
            self._node_id_reverse = {nid: i for i, (nid, _) in enumerate(embedded)}
        else:
            embedded = [(nid, n.get_embedding(self.compressor)) for nid, n in self.nodes.items() if n.get_embedding(self.compressor) is not None]
            if not embedded:
                return

            dim = embedded[0][1].shape[0]
            embs = np.array([emb for _, emb in embedded], dtype=np.float32)

            self._node_index = faiss.IndexFlatIP(dim)
            self._node_index.add(embs)
            self._node_id_map = {i: nid for i, (nid, _) in enumerate(embedded)}
            self._node_id_reverse = {nid: i for i, (nid, _) in enumerate(embedded)}

    # ── Activation ────────────────────────────────────────────────

    def activate_columns(self, query_embedding: np.ndarray, top_k: int = 3) -> list[str]:
        """Find the most relevant cortical columns for a query."""
        if self._column_index is None or self._column_index.ntotal == 0:
            return list(self.columns.keys())

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(top_k, self._column_index.ntotal)
        scores, indices = self._column_index.search(query_embedding.astype(np.float32), k)

        activated = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            col_id = self._column_id_map.get(int(idx))
            if col_id:
                self.columns[col_id].activation = float(score)
                activated.append(col_id)

        return activated

    def find_nearest_nodes(
        self, query_embedding: np.ndarray, top_k: int = 15
    ) -> list[tuple[str, float]]:
        """FAISS search for nearest nodes."""
        if self._node_index is None or self._node_index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if self.compressor is not None:
            # Compress query to binary via TurboQuant model
            _, query_qjl = self.compressor.compress(query_embedding)
            
            # Binary distances are Hamming (smaller is better). 
            # We fetch top_k * 5 candidates to perfectly rerank using precise float math
            fetch_k = min(top_k * 5, self._node_index.ntotal)
            scores, indices = self._node_index.search(query_qjl, fetch_k)
            
            candidates = []
            for idx in indices[0]:
                if idx == -1: continue
                nid = self._node_id_map.get(int(idx))
                if nid:
                    node_emb = self.nodes[nid].get_embedding(self.compressor)
                    if node_emb is not None:
                        # Exact Cosine Similarity for Re-ranking back to absolute precision
                        sim = float(np.dot(query_embedding[0], node_emb) / 
                                   (np.linalg.norm(query_embedding[0]) * np.linalg.norm(node_emb) + 1e-8))
                        candidates.append((nid, sim))
            
            # Sort descending by re-ranked float32 accuracy
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:top_k]
        else:
            k = min(top_k, self._node_index.ntotal)
            scores, indices = self._node_index.search(query_embedding.astype(np.float32), k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                nid = self._node_id_map.get(int(idx))
                if nid:
                    results.append((nid, float(score)))
            return results

    def reset_activations(self) -> None:
        """Reset all transient activations."""
        for node in self.nodes.values():
            node.activation = 0.0
        for col in self.columns.values():
            col.activation = 0.0

    # ── Hebbian Learning ──────────────────────────────────────────

    def hebbian_update(
        self, activated_nodes: dict[str, float], learning_rate: float = settings.cortex_hebbian_lr
    ) -> int:
        """
        Hebbian learning: strengthen synapses between co-activated nodes.
        "Neurons that fire together wire together."

        Returns the number of synapses strengthened.
        """
        updated = 0
        active_ids = list(activated_nodes.keys())

        for i, src_id in enumerate(active_ids):
            src_act = activated_nodes[src_id]
            for tgt_id in active_ids[i + 1:]:
                tgt_act = activated_nodes[tgt_id]

                # Strengthen existing synapse or create a new one
                for s, t in [(src_id, tgt_id), (tgt_id, src_id)]:
                    if self.graph.has_edge(s, t):
                        data = self.graph.edges[s, t]
                        delta = learning_rate * src_act * tgt_act
                        data["weight"] = min(1.0, data["weight"] + delta)
                        data["co_fire_count"] = data.get("co_fire_count", 0) + 1
                        data["last_fired"] = datetime.utcnow().isoformat()
                        updated += 1

        return updated

    # ── Consolidation Engine ──────────────────────────────────────

    def consolidate(self) -> dict[str, int]:
        """
        Memory consolidation — like the brain during sleep:
        1. Decay unused synapses
        2. Prune weak synapses
        3. Merge highly similar nodes
        4. Recompute column centroids

        Returns stats about what changed.
        """
        stats = {"decayed": 0, "pruned": 0, "merged": 0}

        # Step 1: Synaptic decay — weaken unused connections
        for src, tgt, data in list(self.graph.edges(data=True)):
            data["weight"] = max(0.0, data["weight"] * (1.0 - settings.cortex_decay_rate))
            stats["decayed"] += 1

        # Step 2: Synaptic pruning — remove very weak connections
        edges_to_remove = []
        for src, tgt, data in self.graph.edges(data=True):
            if data["weight"] < settings.cortex_pruning_threshold:
                edges_to_remove.append((src, tgt))
        for src, tgt in edges_to_remove:
            self.graph.remove_edge(src, tgt)
        stats["pruned"] = len(edges_to_remove)

        # Step 3: Memory consolidation — merge highly similar, co-accessed nodes
        stats["merged"] = self._merge_similar_nodes()

        # Step 4: Recompute column centroids
        emb_dict = {nid: n.get_embedding(self.compressor) for nid, n in self.nodes.items() if n.get_embedding(self.compressor) is not None}
        for col in self.columns.values():
            col.member_ids = [mid for mid in col.member_ids if mid in self.nodes]
            col.update_centroid(emb_dict)
        self._build_column_index()

        logger.info(f"Consolidation: {stats}")
        return stats

    def _merge_similar_nodes(self) -> int:
        """Merge nodes with very high similarity that are frequently co-accessed."""
        merged = 0
        to_merge: list[tuple[str, str]] = []

        chunk_nodes = [
            (nid, n) for nid, n in self.nodes.items()
            if n.get_embedding(self.compressor) is not None and n.node_type == "chunk"
        ]

        for i, (id_a, node_a) in enumerate(chunk_nodes):
            for id_b, node_b in chunk_nodes[i + 1:]:
                sim = float(np.dot(node_a.get_embedding(self.compressor), node_b.get_embedding(self.compressor)))
                if sim >= settings.cortex_merge_threshold:
                    # Merge b into a (keep the one accessed more)
                    if node_a.access_count >= node_b.access_count:
                        to_merge.append((id_a, id_b))
                    else:
                        to_merge.append((id_b, id_a))

        already_merged = set()
        for keep_id, remove_id in to_merge:
            if remove_id in already_merged or keep_id in already_merged:
                continue

            keep = self.nodes[keep_id]
            remove = self.nodes[remove_id]

            # Consolidate: combine text, merge entities, average embeddings
            keep.text = keep.text + "\n\n" + remove.text
            keep.entity_names = list(set(keep.entity_names + remove.entity_names))
            keep.access_count += remove.access_count
            
            k_emb = keep.get_embedding(self.compressor)
            r_emb = remove.get_embedding(self.compressor)
            if k_emb is not None and r_emb is not None:
                combined = (k_emb + r_emb) / 2.0
                keep.embedding = combined / (np.linalg.norm(combined) + 1e-8)
                # Clear quantized forms if any, since we merged them
                keep.polar_q = None
                keep.qjl_signs = None
            keep.node_type = "consolidated"

            # Redirect all edges from removed node to kept node
            for src, _, data in list(self.graph.in_edges(remove_id, data=True)):
                if src != keep_id:
                    self.graph.add_edge(src, keep_id, **data)
            for _, tgt, data in list(self.graph.out_edges(remove_id, data=True)):
                if tgt != keep_id:
                    self.graph.add_edge(keep_id, tgt, **data)

            # Remove the merged node
            self.graph.remove_node(remove_id)
            del self.nodes[remove_id]
            already_merged.add(remove_id)
            merged += 1

        return merged

    def maybe_consolidate(self) -> Optional[dict]:
        """Trigger consolidation if enough queries have elapsed."""
        self._query_count += 1
        if self._query_count % settings.cortex_consolidation_interval == 0:
            return self.consolidate()
        return None

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return cortex statistics."""
        synapse_types = {}
        for _, _, data in self.graph.edges(data=True):
            st = data.get("synapse_type", "unknown")
            synapse_types[st] = synapse_types.get(st, 0) + 1

        node_types = {}
        for n in self.nodes.values():
            node_types[n.node_type] = node_types.get(n.node_type, 0) + 1

        total_access = sum(n.access_count for n in self.nodes.values())
        avg_weight = 0.0
        edge_count = self.graph.number_of_edges()
        if edge_count > 0:
            avg_weight = sum(d["weight"] for _, _, d in self.graph.edges(data=True)) / edge_count

        return {
            "total_nodes": len(self.nodes),
            "total_synapses": edge_count,
            "total_columns": len(self.columns),
            "node_types": node_types,
            "synapse_types": synapse_types,
            "total_retrievals": total_access,
            "avg_synapse_weight": round(avg_weight, 4),
            "queries_since_consolidation": self._query_count % settings.cortex_consolidation_interval,
            "working_memory_active": len(self.working_memory.active_nodes),
        }

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        nodes = [n.to_dict() for n in self.nodes.values()]
        synapses = []
        for src, tgt, data in self.graph.edges(data=True):
            synapses.append(Synapse(source_id=src, target_id=tgt, **data).to_dict())
        columns = [c.to_dict() for c in self.columns.values()]
        return {
            "nodes": nodes,
            "synapses": synapses,
            "columns": columns,
            "metadata": self.metadata,
            "query_count": self._query_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CortexStore:
        cs = cls()
        cs.metadata = data.get("metadata", {})
        cs._query_count = data.get("query_count", 0)
        for nd in data.get("nodes", []):
            cs.add_node(CortexNode.from_dict(nd))
        for sd in data.get("synapses", []):
            cs.add_synapse(Synapse.from_dict(sd))
        for cd in data.get("columns", []):
            col = CorticalColumn.from_dict(cd)
            cs.columns[col.column_id] = col
        return cs

    def save(self, path: Optional[Path] = None) -> None:
        """Persist the cortex to disk."""
        path = path or settings.cortex_persist_path
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "cortex.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        # Embeddings
        emb_data = {}
        for nid, n in self.nodes.items():
            if n.embedding is not None:
                emb_data[nid] = n.embedding
            elif n.polar_q is not None and n.qjl_signs is not None:
                emb_data[nid] = {"pq": n.polar_q, "qjl": n.qjl_signs}

        with open(path / "embeddings.pkl", "wb") as f:
            pickle.dump(emb_data, f)

        # Column centroids
        centroid_data = {
            cid: c.centroid for cid, c in self.columns.items() if c.centroid is not None
        }
        with open(path / "centroids.pkl", "wb") as f:
            pickle.dump(centroid_data, f)

        # FAISS indices
        if self._node_index:
            faiss.write_index(self._node_index, str(path / "nodes.faiss"))
        if self._column_index:
            faiss.write_index(self._column_index, str(path / "columns.faiss"))

        # ID mappings
        with open(path / "id_maps.json", "w") as f:
            json.dump({
                "node_id_map": {str(k): v for k, v in self._node_id_map.items()},
                "node_id_reverse": self._node_id_reverse,
                "column_id_map": {str(k): v for k, v in self._column_id_map.items()},
            }, f)

        logger.info(f"Saved CortexStore to {path}: {self.stats()}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> CortexStore:
        """Load cortex from disk."""
        path = path or settings.cortex_persist_path

        with open(path / "cortex.json") as f:
            data = json.load(f)
        cs = cls.from_dict(data)

        # Embeddings
        emb_path = path / "embeddings.pkl"
        if emb_path.exists():
            with open(emb_path, "rb") as f:
                emb_data = pickle.load(f)
            for nid, emb in emb_data.items():
                if nid in cs.nodes:
                    if isinstance(emb, dict) and "pq" in emb and "qjl" in emb:
                        cs.nodes[nid].polar_q = emb["pq"]
                        cs.nodes[nid].qjl_signs = emb["qjl"]
                    else:
                        cs.nodes[nid].embedding = emb

        # Column centroids
        centroid_path = path / "centroids.pkl"
        if centroid_path.exists():
            with open(centroid_path, "rb") as f:
                centroid_data = pickle.load(f)
            for cid, centroid in centroid_data.items():
                if cid in cs.columns:
                    cs.columns[cid].centroid = centroid

        # FAISS indices
        node_faiss = path / "nodes.faiss"
        if node_faiss.exists():
            cs._node_index = faiss.read_index(str(node_faiss))
        col_faiss = path / "columns.faiss"
        if col_faiss.exists():
            cs._column_index = faiss.read_index(str(col_faiss))

        # ID mappings
        map_path = path / "id_maps.json"
        if map_path.exists():
            with open(map_path) as f:
                maps = json.load(f)
            cs._node_id_map = {int(k): v for k, v in maps.get("node_id_map", {}).items()}
            cs._node_id_reverse = maps.get("node_id_reverse", {})
            cs._column_id_map = {int(k): v for k, v in maps.get("column_id_map", {}).items()}

        logger.info(f"Loaded CortexStore from {path}: {cs.stats()}")
        return cs


# ── CortexStoreBuilder ───────────────────────────────────────────────


class CortexStoreBuilder:
    """
    Builds a CortexStore from chunks, entities, relationships, and tree.
    Creates 4 synapse types + cortical columns + FAISS indices.
    """

    def __init__(self):
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

    def build(self, chunks, entities, relationships, tree=None) -> CortexStore:
        """Build the full CortexStore from all ingestion data."""
        cs = CortexStore()
        cs.metadata = {
            "built_at": datetime.utcnow().isoformat(),
            "num_chunks": len(chunks),
            "num_entities": len(entities),
            "num_relationships": len(relationships),
        }

        logger.info("Building CortexStore...")

        # Step 1: Chunk nodes
        chunk_texts = [c.text for c in chunks]
        chunk_embs = self._embed(chunk_texts) if chunk_texts else np.array([])
        
        if len(chunk_embs) > 0 and cs.compressor is not None:
            chunk_polar_qs, chunk_qjl_signs = cs.compressor.compress(chunk_embs)
        else:
            chunk_polar_qs, chunk_qjl_signs = None, None

        for i, chunk in enumerate(chunks):
            chunk_entities = [
                e.name for e in entities if chunk.chunk_id in e.source_chunks
            ]
            node = CortexNode(
                node_id=chunk.chunk_id,
                text=chunk.text,
                node_type="chunk",
                entity_names=chunk_entities,
                metadata={"source": chunk.metadata.source, "section": chunk.metadata.section_header},
            )
            # Assign compressed vs float representations
            if chunk_polar_qs is not None:
                node.polar_q = chunk_polar_qs[i]
                node.qjl_signs = chunk_qjl_signs[i]
            elif len(chunk_embs) > 0:
                node.embedding = chunk_embs[i]
                
            cs.add_node(node)

        # Step 2: Entity nodes
        ent_texts = [f"{e.name}: {e.description}" for e in entities]
        ent_embs = self._embed(ent_texts) if ent_texts else np.array([])
        
        if len(ent_embs) > 0 and cs.compressor is not None:
            ent_polar_qs, ent_qjl_signs = cs.compressor.compress(ent_embs)
        else:
            ent_polar_qs, ent_qjl_signs = None, None

        for i, ent in enumerate(entities):
            node = CortexNode(
                node_id=f"entity_{ent.name.lower().replace(' ', '_')}",
                text=f"{ent.name} ({ent.entity_type}): {ent.description}",
                node_type="entity",
                entity_names=[ent.name],
                metadata={"entity_type": ent.entity_type},
            )
            if ent_polar_qs is not None:
                node.polar_q = ent_polar_qs[i]
                node.qjl_signs = ent_qjl_signs[i]
            elif len(ent_embs) > 0:
                node.embedding = ent_embs[i]
                
            cs.add_node(node)

        # Step 3: Tree summary nodes
        if tree:
            for nid, tnode in tree.nodes.items():
                if tnode.level > 0:
                    emb = (
                        np.array(tnode.embedding, dtype=np.float32) if tnode.embedding
                        else self._embed([tnode.text])[0]
                    )
                    cs.add_node(CortexNode(
                        node_id=nid, text=tnode.text, embedding=emb,
                        node_type="summary", metadata={"level": tnode.level},
                    ))

        # Step 4: Create synapses
        self._create_entity_synapses(cs, entities)
        self._create_relationship_synapses(cs, relationships)
        self._create_semantic_synapses(cs, chunks, chunk_embs)
        self._create_co_occurrence_synapses(cs, entities)
        if tree:
            self._create_hierarchical_synapses(cs, tree)

        # Step 5: Build columns + indices
        cs.build_columns()
        cs.build_node_index()

        logger.info(f"CortexStore built: {cs.stats()}")
        return cs

    def _create_entity_synapses(self, cs, entities) -> None:
        """Entity link synapses: entity→chunk, chunk↔chunk (shared entity)."""
        for ent in entities:
            ent_nid = f"entity_{ent.name.lower().replace(' ', '_')}"
            chunk_ids = [cid for cid in ent.source_chunks if cid in cs.nodes]

            for cid in chunk_ids:
                if ent_nid in cs.nodes:
                    cs.add_synapse(Synapse(
                        source_id=ent_nid, target_id=cid,
                        synapse_type="entity_link", weight=0.8, initial_weight=0.8,
                        relationship_desc=f"Entity '{ent.name}' in chunk",
                    ))

            for i, a in enumerate(chunk_ids):
                for b in chunk_ids[i + 1:]:
                    for s, t in [(a, b), (b, a)]:
                        cs.add_synapse(Synapse(
                            source_id=s, target_id=t,
                            synapse_type="entity_link", weight=0.6, initial_weight=0.6,
                            relationship_desc=f"Shared entity: {ent.name}",
                        ))

    def _create_relationship_synapses(self, cs, relationships) -> None:
        """Synapses from extracted entity-entity relationships."""
        for rel in relationships:
            src = f"entity_{rel.source.lower().replace(' ', '_')}"
            tgt = f"entity_{rel.target.lower().replace(' ', '_')}"
            if src in cs.nodes and tgt in cs.nodes:
                w = rel.confidence * 0.9
                cs.add_synapse(Synapse(
                    source_id=src, target_id=tgt,
                    synapse_type="entity_link", weight=w, initial_weight=w,
                    confidence=rel.confidence,
                    relationship_desc=f"{rel.source} --[{rel.relation_type}]--> {rel.target}",
                ))

    def _create_semantic_synapses(self, cs, chunks, embeddings) -> None:
        """Semantic similarity synapses between high-sim chunks."""
        if len(embeddings) < 2:
            return
        sim_matrix = embeddings @ embeddings.T
        threshold = settings.cortex_merge_threshold * 0.73  # ~0.7 for connectivity

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    w = min((sim - threshold) / (1.0 - threshold) * 0.7, 0.7)
                    for s, t in [(chunks[i].chunk_id, chunks[j].chunk_id),
                                 (chunks[j].chunk_id, chunks[i].chunk_id)]:
                        cs.add_synapse(Synapse(
                            source_id=s, target_id=t,
                            synapse_type="semantic", weight=w, initial_weight=w,
                            confidence=sim,
                            relationship_desc=f"Semantic similarity: {sim:.3f}",
                        ))

    def _create_co_occurrence_synapses(self, cs, entities) -> None:
        """Co-occurrence synapses between entities in the same chunk."""
        chunk_ents: dict[str, list[str]] = {}
        for ent in entities:
            for cid in ent.source_chunks:
                chunk_ents.setdefault(cid, []).append(ent.name)

        for cid, names in chunk_ents.items():
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    sa = f"entity_{a.lower().replace(' ', '_')}"
                    sb = f"entity_{b.lower().replace(' ', '_')}"
                    if sa in cs.nodes and sb in cs.nodes:
                        for s, t in [(sa, sb), (sb, sa)]:
                            cs.add_synapse(Synapse(
                                source_id=s, target_id=t,
                                synapse_type="co_occurrence", weight=0.5, initial_weight=0.5,
                                relationship_desc=f"Co-occur in chunk {cid}",
                            ))

    def _create_hierarchical_synapses(self, cs, tree) -> None:
        """RAPTOR tree parent↔child synapses."""
        for nid, tnode in tree.nodes.items():
            if nid not in cs.nodes:
                continue
            for child_id in tnode.children:
                target = child_id
                if child_id not in cs.nodes:
                    child = tree.nodes.get(child_id)
                    if child and child.source_chunk_id:
                        target = child.source_chunk_id
                if target in cs.nodes:
                    cs.add_synapse(Synapse(
                        source_id=nid, target_id=target,
                        synapse_type="hierarchical", weight=0.75, initial_weight=0.75,
                        relationship_desc="RAPTOR parent→child",
                    ))
                    cs.add_synapse(Synapse(
                        source_id=target, target_id=nid,
                        synapse_type="hierarchical", weight=0.5, initial_weight=0.5,
                        relationship_desc="RAPTOR child→parent",
                    ))

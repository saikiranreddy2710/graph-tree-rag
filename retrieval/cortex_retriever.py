"""
CortexRetriever — Three-phase retrieval mimicking neural processing.

Phase 1: Column Activation — FAISS finds relevant cortical columns
Phase 2: Intra-Column Cascade — Activate best nodes, spread via synapses
Phase 3: Inter-Column Propagation — Strong signals cross column boundaries

After retrieval, Hebbian learning strengthens co-activated synapses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from config.settings import settings
from ingestion.cortex_store import CortexStore

logger = logging.getLogger(__name__)


@dataclass
class CortexRetrievalResult:
    """A retrieval result with full neural provenance."""

    node_id: str
    text: str
    activation: float
    node_type: str
    column_id: str = ""
    source_type: str = "cortex"
    hop_distance: int = 0
    entities_matched: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CortexRetrievalTrace:
    """Trace of how the cortex processed this query."""

    columns_activated: list[str] = field(default_factory=list)
    seed_nodes: list[tuple[str, float]] = field(default_factory=list)
    total_nodes_activated: int = 0
    max_hop_reached: int = 0
    synapses_strengthened: int = 0
    working_memory_size: int = 0
    consolidation_triggered: bool = False


class CortexRetriever:
    """
    Three-phase retrieval that mimics how the brain processes a query:

    1. Cortical columns activate (fast, coarse-grained)
    2. Individual neurons within columns fire (fine-grained)
    3. Activation spreads through synapses to related neurons
    4. Hebbian learning updates synapse weights

    The system literally gets smarter with every query.
    """

    def __init__(
        self,
        cortex: CortexStore,
        decay_factor: float = settings.cortex_decay_factor,
        max_hops: int = settings.cortex_max_hops,
        activation_threshold: float = settings.cortex_activation_threshold,
    ):
        self.cortex = cortex
        self.decay_factor = decay_factor
        self.max_hops = max_hops
        self.activation_threshold = activation_threshold

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = settings.cortex_top_k,
    ) -> tuple[list[CortexRetrievalResult], CortexRetrievalTrace]:
        """Full three-phase cortex retrieval."""
        trace = CortexRetrievalTrace()

        # Reset transient activations
        self.cortex.reset_activations()

        # ── Phase 1: Column Activation ────────────────────────────
        active_columns = self.cortex.activate_columns(query_embedding, top_k=3)
        trace.columns_activated = active_columns

        # ── Phase 2: Intra-Column Node Activation ─────────────────
        # Find seed nodes within activated columns + global FAISS search
        seed_nodes = self._activate_seeds(query_embedding, active_columns)
        trace.seed_nodes = [(nid, act) for nid, act in seed_nodes.items()]

        if not seed_nodes:
            return [], trace

        # Set initial activations and track provenance
        hop_distance: dict[str, int] = {}
        node_column: dict[str, str] = {}

        for nid, act in seed_nodes.items():
            self.cortex.nodes[nid].fire(act)
            hop_distance[nid] = 0
            # Determine which column this node belongs to
            for col_id, col in self.cortex.columns.items():
                if nid in col.member_ids:
                    node_column[nid] = col_id
                    break

        # ── Phase 3: Synaptic Propagation ─────────────────────────
        for hop in range(1, self.max_hops + 1):
            decay = self.decay_factor ** hop

            # Get nodes active from previous hops
            active = [
                (nid, self.cortex.nodes[nid].activation)
                for nid in hop_distance
                if hop_distance[nid] == hop - 1
                and self.cortex.nodes[nid].activation > self.activation_threshold
            ]

            new_activations = 0
            for src_id, src_act in active:
                neighbors = self.cortex.get_neighbors(src_id)

                for neighbor_id, edge_data in neighbors:
                    if neighbor_id not in self.cortex.nodes:
                        continue

                    weight = edge_data.get("weight", 0.5)
                    confidence = edge_data.get("confidence", 1.0)

                    # Propagation: activation × synapse_weight × decay
                    propagated = src_act * weight * confidence * decay

                    # Working memory boost — nodes in working memory get extra activation
                    wm_boost = self.cortex.working_memory.get_context_boost(neighbor_id)
                    propagated += wm_boost

                    if propagated > self.activation_threshold:
                        node = self.cortex.nodes[neighbor_id]
                        node.activation = max(node.activation, node.activation + propagated)

                        if neighbor_id not in hop_distance:
                            hop_distance[neighbor_id] = hop
                            node.fire(node.activation)
                            new_activations += 1

                            for col_id, col in self.cortex.columns.items():
                                if neighbor_id in col.member_ids:
                                    node_column[neighbor_id] = col_id
                                    break

            if new_activations > 0:
                trace.max_hop_reached = hop

        # ── Collect Results ───────────────────────────────────────
        activated_nodes = [
            (nid, self.cortex.nodes[nid])
            for nid in hop_distance
            if self.cortex.nodes[nid].activation > self.activation_threshold
        ]
        activated_nodes.sort(key=lambda x: x[1].activation, reverse=True)

        results = []
        for nid, node in activated_nodes[:top_k]:
            results.append(CortexRetrievalResult(
                node_id=nid,
                text=node.text,
                activation=node.activation,
                node_type=node.node_type,
                column_id=node_column.get(nid, ""),
                hop_distance=hop_distance.get(nid, 0),
                entities_matched=node.entity_names,
                metadata=node.metadata,
            ))

        trace.total_nodes_activated = len(activated_nodes)

        # ── Post-Retrieval: Hebbian Learning ──────────────────────
        activated_dict = {nid: node.activation for nid, node in activated_nodes[:top_k]}
        trace.synapses_strengthened = self.cortex.hebbian_update(activated_dict)

        # ── Post-Retrieval: Update Working Memory ─────────────────
        emb_dict = {
            nid: self.cortex.nodes[nid].embedding
            for nid in activated_dict
            if self.cortex.nodes[nid].embedding is not None
        }
        self.cortex.working_memory.update(activated_dict, emb_dict)
        trace.working_memory_size = len(self.cortex.working_memory.active_nodes)

        # ── Maybe Consolidate ─────────────────────────────────────
        consolidation = self.cortex.maybe_consolidate()
        if consolidation is not None:
            trace.consolidation_triggered = True

        logger.info(
            f"Cortex retrieval: {len(trace.columns_activated)} columns → "
            f"{len(seed_nodes)} seeds → {trace.total_nodes_activated} activated → "
            f"{len(results)} returned | "
            f"{trace.synapses_strengthened} synapses strengthened"
        )

        return results, trace

    def _activate_seeds(
        self, query_embedding: np.ndarray, active_columns: list[str]
    ) -> dict[str, float]:
        """Find seed nodes: best nodes within active columns + global FAISS."""
        seeds: dict[str, float] = {}

        # Method 1: FAISS global search
        global_hits = self.cortex.find_nearest_nodes(
            query_embedding, top_k=settings.cortex_top_k
        )
        for nid, score in global_hits:
            seeds[nid] = max(seeds.get(nid, 0), score)

        # Method 2: Best nodes within each active column
        if query_embedding.ndim == 1:
            qe = query_embedding
        else:
            qe = query_embedding[0]

        for col_id in active_columns:
            if col_id not in self.cortex.columns:
                continue
            col = self.cortex.columns[col_id]
            for member_id in col.member_ids:
                node = self.cortex.nodes.get(member_id)
                if node and node.embedding is not None:
                    sim = float(np.dot(qe, node.embedding))
                    if sim > self.activation_threshold:
                        seeds[member_id] = max(seeds.get(member_id, 0), sim)

        return seeds

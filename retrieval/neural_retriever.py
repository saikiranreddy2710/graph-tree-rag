"""
Neural Retriever — Spreading activation retrieval over the NeuralGraph.

Inspired by the brain's memory retrieval: activating one concept
naturally spreads signal to related concepts through synaptic connections.

Algorithm:
  1. Embed query → FAISS search → find seed nodes (initial activation)
  2. Propagate activation through edges: a(neighbor) += a(node) × weight × decay
  3. Repeat for max_hops iterations
  4. Collect all nodes above activation threshold, sorted by activation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from config.settings import settings
from ingestion.neural_graph import NeuralGraph

logger = logging.getLogger(__name__)


@dataclass
class NeuralRetrievalResult:
    """A retrieval result from the neural graph with full provenance."""

    node_id: str
    text: str
    activation: float
    node_type: str
    source_type: str = "neural_graph"
    hop_distance: int = 0  # 0 = direct seed, 1+ = propagated
    activation_path: list[str] = field(default_factory=list)
    entities_matched: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivationTrace:
    """Trace of how activation spread through the graph."""

    seed_nodes: list[tuple[str, float]] = field(default_factory=list)
    propagation_steps: list[dict] = field(default_factory=list)
    total_nodes_activated: int = 0
    max_hop_reached: int = 0


class SpreadingActivationRetriever:
    """
    Retrieves from the NeuralGraph using spreading activation.

    Like how the brain retrieves memories:
    - A stimulus (query) activates nearby neurons (FAISS search)
    - Activation spreads through synapses (graph edges)
    - Stronger connections carry more signal
    - Signal decays with distance
    - The most activated neurons form the retrieved context
    """

    def __init__(
        self,
        neural_graph: NeuralGraph,
        decay_factor: float = settings.neural_graph_decay_factor,
        max_hops: int = settings.neural_graph_max_hops,
        activation_threshold: float = settings.neural_graph_activation_threshold,
        initial_top_k: int = settings.neural_graph_initial_top_k,
    ):
        self.ng = neural_graph
        self.decay_factor = decay_factor
        self.max_hops = max_hops
        self.activation_threshold = activation_threshold
        self.initial_top_k = initial_top_k

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = settings.neural_graph_top_k,
    ) -> tuple[list[NeuralRetrievalResult], ActivationTrace]:
        """
        Full spreading activation retrieval.

        Returns (results, trace) where results are ranked by activation
        and trace shows how activation propagated.
        """
        trace = ActivationTrace()

        # Reset all activations
        self.ng.reset_activations()

        # Step 1: Initial activation — FAISS finds seed nodes
        seed_nodes = self.ng.find_nearest_nodes(query_embedding, self.initial_top_k)
        trace.seed_nodes = seed_nodes

        if not seed_nodes:
            return [], trace

        # Set initial activations from FAISS similarity scores
        node_hop_distance: dict[str, int] = {}
        node_activation_path: dict[str, list[str]] = {}

        for node_id, score in seed_nodes:
            if node_id in self.ng.nodes:
                self.ng.nodes[node_id].activation = max(0.0, score)
                node_hop_distance[node_id] = 0
                node_activation_path[node_id] = [node_id]

        # Step 2: Spreading activation — propagate through edges
        for hop in range(1, self.max_hops + 1):
            decay = self.decay_factor ** hop
            propagation_step = {"hop": hop, "decay": decay, "nodes_activated": 0}

            # Collect nodes to propagate from (those activated in previous hops)
            active_nodes = [
                (nid, node.activation)
                for nid, node in self.ng.nodes.items()
                if node.activation > self.activation_threshold
                and node_hop_distance.get(nid, hop) < hop
            ]

            for source_id, source_activation in active_nodes:
                neighbors = self.ng.get_neighbors(source_id)

                for neighbor_id, edge_data in neighbors:
                    if neighbor_id not in self.ng.nodes:
                        continue

                    edge_weight = edge_data.get("weight", 0.5)
                    edge_confidence = edge_data.get("confidence", 1.0)

                    # Spreading activation formula
                    propagated = source_activation * edge_weight * edge_confidence * decay

                    if propagated > self.activation_threshold:
                        current = self.ng.nodes[neighbor_id].activation
                        # Accumulate activation (like multiple synapses firing)
                        new_activation = current + propagated
                        self.ng.nodes[neighbor_id].activation = new_activation

                        # Track shortest path
                        if neighbor_id not in node_hop_distance:
                            node_hop_distance[neighbor_id] = hop
                            node_activation_path[neighbor_id] = (
                                node_activation_path.get(source_id, [source_id]) + [neighbor_id]
                            )
                            propagation_step["nodes_activated"] += 1

            trace.propagation_steps.append(propagation_step)
            if propagation_step["nodes_activated"] > 0:
                trace.max_hop_reached = hop

        # Step 3: Collect results
        activated_nodes = [
            (nid, node)
            for nid, node in self.ng.nodes.items()
            if node.activation > self.activation_threshold
        ]

        # Sort by activation (highest first)
        activated_nodes.sort(key=lambda x: x[1].activation, reverse=True)

        results = []
        for node_id, node in activated_nodes[:top_k]:
            results.append(
                NeuralRetrievalResult(
                    node_id=node_id,
                    text=node.text,
                    activation=node.activation,
                    node_type=node.node_type,
                    hop_distance=node_hop_distance.get(node_id, 0),
                    activation_path=node_activation_path.get(node_id, [node_id]),
                    entities_matched=node.entity_names,
                    metadata=node.metadata,
                )
            )

        trace.total_nodes_activated = len(activated_nodes)

        logger.info(
            f"Neural retrieval: {len(seed_nodes)} seeds → "
            f"{trace.total_nodes_activated} activated → "
            f"{len(results)} returned (max hop: {trace.max_hop_reached})"
        )

        return results, trace

"""Tests for neural graph construction, serialization, and spreading activation retrieval."""

import numpy as np
import pytest

from ingestion.neural_graph import NeuralEdge, NeuralGraph, NeuralGraphBuilder, NeuralNode
from retrieval.neural_retriever import SpreadingActivationRetriever


# ── Helper factories ──────────────────────────────────────────────────


def _make_node(node_id: str, text: str = "test text", node_type: str = "chunk", dim: int = 8):
    """Create a NeuralNode with a random normalized embedding."""
    emb = np.random.randn(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return NeuralNode(
        node_id=node_id,
        text=text,
        embedding=emb,
        node_type=node_type,
        entity_names=[],
    )


def _build_simple_graph() -> NeuralGraph:
    """
    Build a small graph with known topology for testing:

        A ──(0.8)──> B ──(0.6)──> C ──(0.5)──> D
                     │
                     └──(0.4)──> E
    """
    ng = NeuralGraph()

    # Use embeddings that make A the closest to a target query
    dim = 8
    query_direction = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    nodes = {
        "A": np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "B": np.array([0.5, 0.5, 0.5, 0, 0, 0, 0, 0], dtype=np.float32),
        "C": np.array([0.1, 0.1, 0.9, 0, 0, 0, 0, 0], dtype=np.float32),
        "D": np.array([0, 0, 0, 0.9, 0, 0, 0, 0], dtype=np.float32),
        "E": np.array([0, 0, 0, 0, 0.9, 0, 0, 0], dtype=np.float32),
    }

    for name, emb in nodes.items():
        emb_norm = emb / np.linalg.norm(emb)
        ng.add_node(NeuralNode(
            node_id=name,
            text=f"Content of node {name}",
            embedding=emb_norm,
            node_type="chunk",
            entity_names=[f"entity_{name}"],
        ))

    edges = [
        ("A", "B", "entity_link", 0.8),
        ("B", "C", "semantic_similarity", 0.6),
        ("C", "D", "co_occurrence", 0.5),
        ("B", "E", "hierarchical", 0.4),
    ]
    for src, tgt, etype, weight in edges:
        ng.add_edge(NeuralEdge(
            source_id=src, target_id=tgt, edge_type=etype, weight=weight,
        ))

    ng.build_faiss_index()
    return ng


# ── NeuralNode Tests ──────────────────────────────────────────────────


class TestNeuralNode:
    def test_creation(self):
        node = _make_node("test_1", text="Hello world")
        assert node.node_id == "test_1"
        assert node.text == "Hello world"
        assert node.embedding is not None
        assert node.embedding.shape == (8,)
        assert node.activation == 0.0

    def test_serialization_roundtrip(self):
        node = _make_node("test_2", text="Some text", node_type="entity")
        d = node.to_dict()
        assert d["node_id"] == "test_2"
        assert d["node_type"] == "entity"

        restored = NeuralNode.from_dict(d)
        assert restored.node_id == node.node_id
        assert restored.text == node.text
        assert restored.node_type == node.node_type


# ── NeuralEdge Tests ──────────────────────────────────────────────────


class TestNeuralEdge:
    def test_creation(self):
        edge = NeuralEdge(
            source_id="a", target_id="b", edge_type="entity_link",
            weight=0.75, relationship_desc="shared entity"
        )
        assert edge.source_id == "a"
        assert edge.weight == 0.75
        assert edge.edge_type == "entity_link"

    def test_serialization_roundtrip(self):
        edge = NeuralEdge(
            source_id="x", target_id="y", edge_type="semantic_similarity",
            weight=0.6, confidence=0.9
        )
        d = edge.to_dict()
        restored = NeuralEdge.from_dict(d)
        assert restored.source_id == edge.source_id
        assert restored.weight == edge.weight
        assert restored.edge_type == edge.edge_type


# ── NeuralGraph Tests ─────────────────────────────────────────────────


class TestNeuralGraph:
    def test_add_node(self):
        ng = NeuralGraph()
        node = _make_node("n1")
        ng.add_node(node)
        assert "n1" in ng.nodes
        assert ng.graph.has_node("n1")

    def test_add_edge(self):
        ng = NeuralGraph()
        ng.add_node(_make_node("a"))
        ng.add_node(_make_node("b"))
        ng.add_edge(NeuralEdge(source_id="a", target_id="b", edge_type="entity_link", weight=0.5))
        assert ng.graph.has_edge("a", "b")

    def test_add_edge_missing_node_ignored(self):
        ng = NeuralGraph()
        ng.add_node(_make_node("a"))
        ng.add_edge(NeuralEdge(source_id="a", target_id="missing", edge_type="entity_link"))
        assert ng.graph.number_of_edges() == 0

    def test_get_neighbors(self):
        ng = _build_simple_graph()
        neighbors = ng.get_neighbors("B")
        neighbor_ids = [n_id for n_id, _ in neighbors]
        assert "C" in neighbor_ids  # outgoing
        assert "E" in neighbor_ids  # outgoing
        assert "A" in neighbor_ids  # incoming

    def test_faiss_index_search(self):
        ng = _build_simple_graph()
        # Query most similar to node A
        query = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        query /= np.linalg.norm(query)
        results = ng.find_nearest_nodes(query, top_k=3)
        assert len(results) >= 1
        # A should be the closest match
        assert results[0][0] == "A"

    def test_reset_activations(self):
        ng = _build_simple_graph()
        ng.nodes["A"].activation = 0.9
        ng.nodes["B"].activation = 0.5
        ng.reset_activations()
        assert all(n.activation == 0.0 for n in ng.nodes.values())

    def test_stats(self):
        ng = _build_simple_graph()
        stats = ng.stats()
        assert stats["total_nodes"] == 5
        assert stats["total_edges"] == 4
        assert "entity_link" in stats["edge_types"]

    def test_graph_serialization_roundtrip(self):
        ng = _build_simple_graph()
        d = ng.to_dict()
        restored = NeuralGraph.from_dict(d)
        assert len(restored.nodes) == len(ng.nodes)
        assert restored.graph.number_of_edges() == ng.graph.number_of_edges()
        for node_id in ng.nodes:
            assert node_id in restored.nodes


# ── Spreading Activation Retriever Tests ──────────────────────────────


class TestSpreadingActivationRetriever:
    def setup_method(self):
        self.ng = _build_simple_graph()

    def test_basic_retrieval(self):
        retriever = SpreadingActivationRetriever(
            self.ng, decay_factor=0.7, max_hops=2,
            activation_threshold=0.01, initial_top_k=5
        )
        # Query closest to node A
        query = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        query /= np.linalg.norm(query)
        results, trace = retriever.retrieve(query, top_k=10)
        assert len(results) > 0
        # A should be among the results (direct seed)
        result_ids = [r.node_id for r in results]
        assert "A" in result_ids

    def test_propagation_reaches_neighbors(self):
        retriever = SpreadingActivationRetriever(
            self.ng, decay_factor=0.9, max_hops=3,
            activation_threshold=0.001, initial_top_k=5
        )
        query = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        query /= np.linalg.norm(query)
        results, trace = retriever.retrieve(query, top_k=10)
        result_ids = [r.node_id for r in results]
        # B should be activated via A→B edge
        assert "B" in result_ids

    def test_activation_decay_with_distance(self):
        retriever = SpreadingActivationRetriever(
            self.ng, decay_factor=0.5, max_hops=3,
            activation_threshold=0.001, initial_top_k=5
        )
        query = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        query /= np.linalg.norm(query)
        results, trace = retriever.retrieve(query, top_k=10)

        result_map = {r.node_id: r for r in results}
        if "A" in result_map and "B" in result_map:
            # A (seed) should have higher activation than B (1-hop)
            assert result_map["A"].activation > result_map["B"].activation

    def test_activation_threshold_filters(self):
        retriever = SpreadingActivationRetriever(
            self.ng, decay_factor=0.1, max_hops=1,
            activation_threshold=0.5, initial_top_k=3
        )
        query = np.array([0, 0, 0, 0, 0.9, 0.1, 0, 0], dtype=np.float32)
        query /= np.linalg.norm(query)
        results, trace = retriever.retrieve(query, top_k=10)
        # With high threshold and low decay, only very close nodes should appear
        for r in results:
            assert r.activation >= 0.5

    def test_trace_has_seed_info(self):
        retriever = SpreadingActivationRetriever(
            self.ng, decay_factor=0.7, max_hops=1,
            activation_threshold=0.01, initial_top_k=3
        )
        query = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        query /= np.linalg.norm(query)
        _, trace = retriever.retrieve(query, top_k=10)
        assert len(trace.seed_nodes) > 0
        assert trace.total_nodes_activated >= 1

    def test_hop_distance_tracking(self):
        retriever = SpreadingActivationRetriever(
            self.ng, decay_factor=0.9, max_hops=3,
            activation_threshold=0.001, initial_top_k=5
        )
        query = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        query /= np.linalg.norm(query)
        results, _ = retriever.retrieve(query, top_k=10)
        result_map = {r.node_id: r for r in results}
        if "A" in result_map:
            assert result_map["A"].hop_distance == 0  # seed
        if "B" in result_map:
            # B could be direct seed OR 1-hop from A
            assert result_map["B"].hop_distance <= 1

    def test_empty_graph(self):
        empty = NeuralGraph()
        retriever = SpreadingActivationRetriever(empty)
        query = np.random.randn(8).astype(np.float32)
        results, trace = retriever.retrieve(query)
        assert results == []
        assert trace.seed_nodes == []

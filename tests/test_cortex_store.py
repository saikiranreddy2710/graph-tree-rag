"""Tests for CortexStore — brain-inspired storage with Hebbian learning."""

import numpy as np
import pytest

from ingestion.cortex_store import CortexNode, CortexStore, Synapse, CorticalColumn, WorkingMemory
from retrieval.cortex_retriever import CortexRetriever


# ── Helpers ───────────────────────────────────────────────────────────


def _make_node(node_id: str, text: str = "test", dim: int = 8) -> CortexNode:
    emb = np.random.randn(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return CortexNode(node_id=node_id, text=text, embedding=emb)


def _build_cortex() -> CortexStore:
    """
    Build a small cortex:
        A ──(0.8)──> B ──(0.6)──> C
                     │
                     └──(0.4)──> D
    """
    cs = CortexStore()
    nodes = {
        "A": np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "B": np.array([0.5, 0.5, 0.5, 0, 0, 0, 0, 0], dtype=np.float32),
        "C": np.array([0.1, 0.1, 0.9, 0, 0, 0, 0, 0], dtype=np.float32),
        "D": np.array([0, 0, 0, 0, 0.9, 0, 0, 0], dtype=np.float32),
    }
    for name, emb in nodes.items():
        emb_norm = emb / np.linalg.norm(emb)
        cs.add_node(CortexNode(
            node_id=name, text=f"Content of {name}", embedding=emb_norm,
            node_type="chunk", entity_names=[f"ent_{name}"],
        ))

    cs.add_synapse(Synapse(source_id="A", target_id="B", synapse_type="entity_link", weight=0.8, initial_weight=0.8))
    cs.add_synapse(Synapse(source_id="B", target_id="C", synapse_type="semantic", weight=0.6, initial_weight=0.6))
    cs.add_synapse(Synapse(source_id="B", target_id="D", synapse_type="co_occurrence", weight=0.4, initial_weight=0.4))

    cs.build_columns(n_columns=2)
    cs.build_node_index()
    return cs


# ── CortexNode Tests ─────────────────────────────────────────────────


class TestCortexNode:
    def test_creation(self):
        node = _make_node("n1")
        assert node.node_id == "n1"
        assert node.activation == 0.0
        assert node.access_count == 0

    def test_fire(self):
        node = _make_node("n1")
        node.fire(0.9)
        assert node.activation == 0.9
        assert node.access_count == 1
        assert node.last_accessed is not None

    def test_serialization(self):
        node = _make_node("n1", text="hello")
        d = node.to_dict()
        restored = CortexNode.from_dict(d)
        assert restored.node_id == "n1"
        assert restored.text == "hello"


# ── Synapse Tests ─────────────────────────────────────────────────────


class TestSynapse:
    def test_creation(self):
        syn = Synapse(source_id="a", target_id="b", synapse_type="entity_link", weight=0.7)
        assert syn.weight == 0.7
        assert syn.co_fire_count == 0

    def test_hebbian_update(self):
        syn = Synapse(source_id="a", target_id="b", synapse_type="entity_link", weight=0.5)
        syn.hebbian_update(src_activation=0.8, tgt_activation=0.6, learning_rate=0.1)
        assert syn.weight > 0.5  # Should have strengthened
        assert syn.co_fire_count == 1
        expected = 0.5 + 0.1 * 0.8 * 0.6  # 0.548
        assert abs(syn.weight - expected) < 1e-6

    def test_decay(self):
        syn = Synapse(source_id="a", target_id="b", synapse_type="entity_link", weight=0.5)
        syn.decay(decay_rate=0.1)
        assert syn.weight == pytest.approx(0.45)

    def test_weight_capped_at_1(self):
        syn = Synapse(source_id="a", target_id="b", synapse_type="entity_link", weight=0.95)
        syn.hebbian_update(src_activation=1.0, tgt_activation=1.0, learning_rate=0.5)
        assert syn.weight == 1.0

    def test_serialization(self):
        syn = Synapse(source_id="x", target_id="y", synapse_type="semantic", weight=0.6)
        d = syn.to_dict()
        restored = Synapse.from_dict(d)
        assert restored.source_id == "x"
        assert restored.weight == 0.6


# ── CortexStore Tests ─────────────────────────────────────────────────


class TestCortexStore:
    def test_add_node(self):
        cs = CortexStore()
        cs.add_node(_make_node("n1"))
        assert "n1" in cs.nodes
        assert cs.graph.has_node("n1")

    def test_add_synapse(self):
        cs = CortexStore()
        cs.add_node(_make_node("a"))
        cs.add_node(_make_node("b"))
        cs.add_synapse(Synapse(source_id="a", target_id="b", synapse_type="entity_link"))
        assert cs.graph.has_edge("a", "b")

    def test_missing_node_synapse_ignored(self):
        cs = CortexStore()
        cs.add_node(_make_node("a"))
        cs.add_synapse(Synapse(source_id="a", target_id="missing", synapse_type="entity_link"))
        assert cs.graph.number_of_edges() == 0

    def test_neighbors(self):
        cs = _build_cortex()
        neighbors = cs.get_neighbors("B")
        ids = [nid for nid, _ in neighbors]
        assert "C" in ids
        assert "D" in ids
        assert "A" in ids  # incoming

    def test_faiss_search(self):
        cs = _build_cortex()
        query = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        query /= np.linalg.norm(query)
        results = cs.find_nearest_nodes(query, top_k=3)
        assert len(results) >= 1
        assert results[0][0] == "A"

    def test_columns_built(self):
        cs = _build_cortex()
        assert len(cs.columns) >= 1

    def test_hebbian_update(self):
        cs = _build_cortex()
        original_weight = cs.graph.edges["A", "B"]["weight"]
        activated = {"A": 0.9, "B": 0.7}
        updated = cs.hebbian_update(activated, learning_rate=0.1)
        new_weight = cs.graph.edges["A", "B"]["weight"]
        assert new_weight > original_weight
        assert updated > 0

    def test_consolidation_decay(self):
        cs = _build_cortex()
        original = cs.graph.edges["A", "B"]["weight"]
        stats = cs.consolidate()
        new = cs.graph.edges["A", "B"]["weight"]
        assert new < original  # Decayed
        assert stats["decayed"] > 0

    def test_consolidation_pruning(self):
        cs = CortexStore()
        cs.add_node(_make_node("x"))
        cs.add_node(_make_node("y"))
        cs.add_synapse(Synapse(source_id="x", target_id="y", synapse_type="entity_link", weight=0.01))
        cs.build_columns(n_columns=1)
        cs.build_node_index()
        stats = cs.consolidate()
        assert stats["pruned"] >= 1
        assert not cs.graph.has_edge("x", "y")

    def test_serialization_roundtrip(self):
        cs = _build_cortex()
        d = cs.to_dict()
        restored = CortexStore.from_dict(d)
        assert len(restored.nodes) == len(cs.nodes)
        assert restored.graph.number_of_edges() == cs.graph.number_of_edges()

    def test_stats(self):
        cs = _build_cortex()
        stats = cs.stats()
        assert stats["total_nodes"] == 4
        assert stats["total_synapses"] == 3
        assert stats["total_columns"] >= 1


# ── WorkingMemory Tests ───────────────────────────────────────────────


class TestWorkingMemory:
    def test_update_and_capacity(self):
        wm = WorkingMemory(capacity=2)
        embs = {"a": np.ones(8, dtype=np.float32), "b": np.ones(8, dtype=np.float32), "c": np.ones(8, dtype=np.float32)}
        wm.update({"a": 0.9, "b": 0.5, "c": 0.3}, embs)
        assert len(wm.active_nodes) == 2  # Capped
        assert "a" in wm.active_nodes
        assert "b" in wm.active_nodes

    def test_context_boost(self):
        wm = WorkingMemory(capacity=5)
        embs = {"a": np.ones(8, dtype=np.float32)}
        wm.update({"a": 0.8}, embs)
        boost = wm.get_context_boost("a")
        assert boost > 0
        assert wm.get_context_boost("missing") == 0

    def test_reset(self):
        wm = WorkingMemory(capacity=5)
        embs = {"a": np.ones(8, dtype=np.float32)}
        wm.update({"a": 0.8}, embs)
        wm.reset()
        assert len(wm.active_nodes) == 0


# ── CortexRetriever Tests ────────────────────────────────────────────


class TestCortexRetriever:
    def setup_method(self):
        self.cs = _build_cortex()

    def test_basic_retrieval(self):
        retriever = CortexRetriever(
            self.cs, decay_factor=0.7, max_hops=2, activation_threshold=0.01
        )
        query = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        query /= np.linalg.norm(query)
        results, trace = retriever.retrieve(query, top_k=10)
        assert len(results) > 0
        ids = [r.node_id for r in results]
        assert "A" in ids

    def test_propagation(self):
        retriever = CortexRetriever(
            self.cs, decay_factor=0.9, max_hops=3, activation_threshold=0.001
        )
        query = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        query /= np.linalg.norm(query)
        results, trace = retriever.retrieve(query, top_k=10)
        ids = [r.node_id for r in results]
        assert "B" in ids  # Should propagate A→B

    def test_hebbian_strengthening_happens(self):
        retriever = CortexRetriever(
            self.cs, decay_factor=0.9, max_hops=2, activation_threshold=0.001
        )
        original = self.cs.graph.edges["A", "B"]["weight"]
        query = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        query /= np.linalg.norm(query)
        results, trace = retriever.retrieve(query, top_k=10)
        ids = [r.node_id for r in results]
        if "A" in ids and "B" in ids:
            new = self.cs.graph.edges["A", "B"]["weight"]
            assert new >= original  # Hebbian strengthened

    def test_trace_info(self):
        retriever = CortexRetriever(
            self.cs, decay_factor=0.7, max_hops=2, activation_threshold=0.01
        )
        query = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        query /= np.linalg.norm(query)
        _, trace = retriever.retrieve(query, top_k=10)
        assert len(trace.columns_activated) >= 1
        assert trace.total_nodes_activated >= 1

    def test_empty_cortex(self):
        empty = CortexStore()
        retriever = CortexRetriever(empty)
        query = np.random.randn(8).astype(np.float32)
        results, trace = retriever.retrieve(query)
        assert results == []

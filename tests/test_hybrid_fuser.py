"""Tests for hybrid fusion engine."""

import pytest
from retrieval.hybrid_fuser import HybridFuser, FusedResult


class TestHybridFuser:
    def setup_method(self):
        self.fuser = HybridFuser()

    def test_empty_inputs(self):
        results = self.fuser.fuse([], [], [], [])
        assert results == []

    def test_single_channel(self):
        vector = [
            {"chunk_id": "a", "text": "doc a", "score": 0.9},
            {"chunk_id": "b", "text": "doc b", "score": 0.7},
        ]
        results = self.fuser.fuse(vector, [], [], [])
        assert len(results) == 2
        assert results[0].doc_id == "a"  # Higher score first

    def test_multi_channel_fusion(self):
        vector = [{"chunk_id": "a", "text": "doc a", "score": 0.9}]
        bm25 = [{"chunk_id": "b", "text": "doc b", "score": 5.0}]
        graph = [
            {
                "chunk_id": "a",
                "text": "doc a",
                "score": 0.8,
                "source_type": "graph_walk",
                "entities_matched": ["X"],
                "metadata": {},
            }
        ]
        tree = [{"node_id": "c", "text": "doc c", "score": 0.85, "metadata": {}}]

        results = self.fuser.fuse(vector, bm25, graph, tree)
        # Doc "a" should be boosted because it appears in 2 channels
        doc_a_results = [r for r in results if r.doc_id == "a"]
        assert len(doc_a_results) == 1
        assert len(doc_a_results[0].channel_scores) >= 2

    def test_top_k_limit(self):
        vector = [
            {"chunk_id": f"doc_{i}", "text": f"text {i}", "score": 1.0 - i * 0.1} for i in range(20)
        ]
        results = self.fuser.fuse(vector, [], [], [], top_k=5)
        assert len(results) == 5

    def test_deduplication(self):
        vector = [{"chunk_id": "a", "text": "doc a", "score": 0.9}]
        bm25 = [{"chunk_id": "a", "text": "doc a longer", "score": 5.0}]
        results = self.fuser.fuse(vector, bm25, [], [])
        assert len(results) == 1
        # Should keep the longer text version
        assert "longer" in results[0].text

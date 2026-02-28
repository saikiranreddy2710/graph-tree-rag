"""Tests for query router."""

import pytest
from config.settings import RetrievalStrategy
from retrieval.query_router import QueryRouter, QueryType


class TestQueryRouter:
    def setup_method(self):
        self.router = QueryRouter()

    def test_simple_query(self):
        strategy, confidence, reasoning = self.router._estimate_complexity("What is Python?")
        assert strategy == RetrievalStrategy.SIMPLE

    def test_comparison_query(self):
        query_type = self.router._classify_query_type(
            "Compare Python and JavaScript for web development"
        )
        assert query_type == QueryType.COMPARISON

    def test_causal_query(self):
        query_type = self.router._classify_query_type(
            "Why does machine learning require large datasets?"
        )
        assert query_type == QueryType.CAUSAL

    def test_temporal_query(self):
        query_type = self.router._classify_query_type(
            "When was the first transformer model introduced?"
        )
        assert query_type == QueryType.TEMPORAL

    def test_complex_query_detection(self):
        strategy, confidence, reasoning = self.router._estimate_complexity(
            "Compare GraphRAG and RAPTOR approaches for document retrieval, "
            "explaining the advantages and disadvantages of each method in "
            "terms of both local and global query performance, and also "
            "discuss how they handle multi-hop reasoning?"
        )
        assert strategy == RetrievalStrategy.COMPLEX
        assert confidence > 0.4

    def test_aggregation_query(self):
        query_type = self.router._classify_query_type("List all the main themes in the dataset")
        assert query_type == QueryType.AGGREGATION

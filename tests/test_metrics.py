"""Tests for evaluation metrics."""

import pytest
from evaluation.metrics import MetricsCalculator


class TestMetricsCalculator:
    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_precision_at_k_perfect(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert self.calc.precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_precision_at_k_none(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert self.calc.precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_precision_at_k_partial(self):
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b", "c"}
        assert self.calc.precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_recall_at_k(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c", "d"}
        assert self.calc.recall_at_k(retrieved, relevant, k=2) == 0.5

    def test_mrr_first(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert self.calc.mrr(retrieved, relevant) == 1.0

    def test_mrr_second(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert self.calc.mrr(retrieved, relevant) == 0.5

    def test_mrr_not_found(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert self.calc.mrr(retrieved, relevant) == 0.0

    def test_ndcg_perfect(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert self.calc.ndcg_at_k(retrieved, relevant, k=3) == 1.0

    def test_word_overlap(self):
        sim = self.calc.word_overlap_similarity("the cat sat on the mat", "the cat on the mat")
        assert sim > 0.8

    def test_word_overlap_no_match(self):
        sim = self.calc.word_overlap_similarity("hello world", "foo bar baz")
        assert sim == 0.0

    def test_entity_coverage_full(self):
        assert self.calc.entity_coverage({"a", "b"}, {"a", "b"}) == 1.0

    def test_entity_coverage_partial(self):
        assert self.calc.entity_coverage({"a"}, {"a", "b"}) == 0.5

    def test_entity_coverage_empty_gold(self):
        assert self.calc.entity_coverage(set(), set()) == 1.0

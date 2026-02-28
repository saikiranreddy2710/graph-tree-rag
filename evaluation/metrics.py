"""
Evaluation Metrics — Comprehensive RAG metrics for retrieval and generation quality.
Implements retrieval metrics (Precision@K, MRR, NDCG) and generation metrics
(faithfulness, answer relevance, BERTScore approximation).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    hit_rate: float = 0.0
    entity_coverage: float = 0.0
    community_hit_rate: float = 0.0
    k: int = 10

    def to_dict(self) -> dict:
        return {
            f"precision@{self.k}": round(self.precision_at_k, 4),
            f"recall@{self.k}": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            f"ndcg@{self.k}": round(self.ndcg, 4),
            "hit_rate": round(self.hit_rate, 4),
            "entity_coverage": round(self.entity_coverage, 4),
            "community_hit_rate": round(self.community_hit_rate, 4),
        }


@dataclass
class GenerationMetrics:
    faithfulness: float = 0.0  # Are claims supported by context?
    answer_relevance: float = 0.0  # Does answer address the query?
    context_precision: float = 0.0  # How precise is the retrieved context?
    answer_similarity: float = 0.0  # Similarity to gold answer (if available)
    citation_accuracy: float = 0.0
    answer_length: int = 0

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_precision": round(self.context_precision, 4),
            "answer_similarity": round(self.answer_similarity, 4),
            "citation_accuracy": round(self.citation_accuracy, 4),
            "answer_length": self.answer_length,
        }


@dataclass
class EvalResult:
    query: str
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict(),
            "metadata": self.metadata,
        }


class MetricsCalculator:
    """Calculates retrieval and generation metrics."""

    @staticmethod
    def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
        """Fraction of retrieved documents that are relevant."""
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_count / len(top_k)

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
        """Fraction of relevant documents that were retrieved."""
        if not relevant_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        found = len(top_k & relevant_ids)
        return found / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        """Mean Reciprocal Rank — 1/(rank of first relevant doc)."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
        """Normalized Discounted Cumulative Gain."""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = 1.0 if doc_id in relevant_ids else 0.0
            dcg += rel / math.log2(i + 2)

        # Ideal DCG
        ideal_rels = min(len(relevant_ids), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def entity_coverage(retrieved_entities: set[str], gold_entities: set[str]) -> float:
        """Fraction of gold entities covered by retrieval."""
        if not gold_entities:
            return 1.0
        found = len(retrieved_entities & gold_entities)
        return found / len(gold_entities)

    @staticmethod
    def word_overlap_similarity(text_a: str, text_b: str) -> float:
        """Simple word overlap similarity (BERTScore approximation)."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        precision = intersection / len(words_a)
        recall = intersection / len(words_b)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def evaluate_retrieval(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        retrieved_entities: set[str] = None,
        gold_entities: set[str] = None,
        k: int = 10,
    ) -> RetrievalMetrics:
        """Calculate all retrieval metrics."""
        return RetrievalMetrics(
            precision_at_k=self.precision_at_k(retrieved_ids, relevant_ids, k),
            recall_at_k=self.recall_at_k(retrieved_ids, relevant_ids, k),
            mrr=self.mrr(retrieved_ids, relevant_ids),
            ndcg=self.ndcg_at_k(retrieved_ids, relevant_ids, k),
            hit_rate=1.0 if any(d in relevant_ids for d in retrieved_ids[:k]) else 0.0,
            entity_coverage=(
                self.entity_coverage(retrieved_entities or set(), gold_entities or set())
            ),
            k=k,
        )

    def evaluate_generation(
        self,
        answer: str,
        gold_answer: Optional[str] = None,
        reflection_scores: Optional[dict] = None,
    ) -> GenerationMetrics:
        """Calculate generation metrics."""
        metrics = GenerationMetrics(answer_length=len(answer))

        if reflection_scores:
            metrics.faithfulness = reflection_scores.get("support", 0.0)
            metrics.answer_relevance = reflection_scores.get("usefulness", 0.0)
            metrics.context_precision = reflection_scores.get("relevance", 0.0)
            metrics.citation_accuracy = reflection_scores.get("citations", 0.0)

        if gold_answer:
            metrics.answer_similarity = self.word_overlap_similarity(answer, gold_answer)

        return metrics

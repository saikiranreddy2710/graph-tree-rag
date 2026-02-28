"""
Failure Analyzer — Automatically categorizes retrieval/generation failures
for iterative improvement.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    WRONG_ENTITY = "wrong_entity"  # Retrieved docs about wrong entity
    MISSING_CONTEXT = "missing_context"  # Relevant info not in corpus
    HALLUCINATION = "hallucination"  # Answer not supported by context
    STALE_DATA = "stale_data"  # Outdated information used
    SECURITY_BLOCK = "security_block"  # Legitimate query blocked
    PARTIAL_ANSWER = "partial_answer"  # Answer incomplete
    NO_RETRIEVAL = "no_retrieval"  # No results found
    WRONG_REASONING = "wrong_reasoning"  # Logic error in answer


@dataclass
class FailureCase:
    query: str
    failure_type: FailureType
    expected: str
    actual: str
    confidence: float = 0.0
    retrieved_doc_ids: list[str] = field(default_factory=list)
    analysis: str = ""
    trace_id: str = ""


class FailureAnalyzer:
    """Analyzes and categorizes pipeline failures."""

    def classify_failure(
        self,
        query: str,
        answer: str,
        gold_answer: str,
        retrieved_ids: list[str],
        reflection_scores: Optional[dict] = None,
        crag_quality: str = "",
    ) -> Optional[FailureCase]:
        """Classify a failure based on available signals."""
        if not gold_answer:
            return None

        # Simple heuristic classification
        scores = reflection_scores or {}
        support = scores.get("support", 1.0)
        relevance = scores.get("relevance", 1.0)
        usefulness = scores.get("usefulness", 1.0)

        # Check if it's actually a failure
        from evaluation.metrics import MetricsCalculator

        calc = MetricsCalculator()
        similarity = calc.word_overlap_similarity(answer, gold_answer)
        if similarity > 0.7:
            return None  # Not a failure

        failure_type = FailureType.PARTIAL_ANSWER  # Default

        if not retrieved_ids:
            failure_type = FailureType.NO_RETRIEVAL
        elif crag_quality == "INCORRECT":
            failure_type = FailureType.WRONG_ENTITY
        elif support < 0.4:
            failure_type = FailureType.HALLUCINATION
        elif relevance < 0.4:
            failure_type = FailureType.MISSING_CONTEXT
        elif usefulness < 0.4:
            failure_type = FailureType.WRONG_REASONING

        return FailureCase(
            query=query,
            failure_type=failure_type,
            expected=gold_answer[:500],
            actual=answer[:500],
            confidence=similarity,
            retrieved_doc_ids=retrieved_ids[:10],
            analysis=f"Similarity: {similarity:.3f}, CRAG: {crag_quality}, "
            f"Support: {support:.2f}, Relevance: {relevance:.2f}",
        )

    def summarize_failures(self, failures: list[FailureCase]) -> dict:
        """Generate a summary of failure patterns."""
        if not failures:
            return {"total": 0, "message": "No failures detected"}

        type_counts = Counter(f.failure_type.value for f in failures)
        total = len(failures)

        return {
            "total_failures": total,
            "by_type": dict(type_counts.most_common()),
            "most_common": type_counts.most_common(1)[0][0] if type_counts else None,
            "avg_confidence": sum(f.confidence for f in failures) / total,
            "recommendations": self._generate_recommendations(type_counts),
        }

    def _generate_recommendations(self, type_counts: Counter) -> list[str]:
        """Generate actionable recommendations based on failure patterns."""
        recs = []
        for ftype, count in type_counts.most_common():
            if ftype == FailureType.WRONG_ENTITY.value:
                recs.append(
                    f"[{count}x] WRONG_ENTITY: Improve entity disambiguation "
                    "in graph retriever. Consider adding entity descriptions to "
                    "the fuzzy matching."
                )
            elif ftype == FailureType.MISSING_CONTEXT.value:
                recs.append(
                    f"[{count}x] MISSING_CONTEXT: Expand corpus or enable web "
                    "search fallback. Check if relevant docs exist but aren't being retrieved."
                )
            elif ftype == FailureType.HALLUCINATION.value:
                recs.append(
                    f"[{count}x] HALLUCINATION: Tighten Self-RAG support threshold. "
                    "Consider increasing CRAG confidence threshold."
                )
            elif ftype == FailureType.NO_RETRIEVAL.value:
                recs.append(
                    f"[{count}x] NO_RETRIEVAL: Check index coverage. Enable HyDE "
                    "for better recall on unusual queries."
                )
            elif ftype == FailureType.WRONG_REASONING.value:
                recs.append(
                    f"[{count}x] WRONG_REASONING: Improve CoT decomposition. "
                    "Consider adding more reasoning strategies to Speculative RAG."
                )
        return recs

    def save_failures(self, failures: list[FailureCase], path: str) -> None:
        """Save failure cases to JSON for review."""
        data = [
            {
                "query": f.query,
                "type": f.failure_type.value,
                "expected": f.expected,
                "actual": f.actual,
                "confidence": f.confidence,
                "analysis": f.analysis,
            }
            for f in failures
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

"""
Adaptive Query Router — Routes queries to the optimal retrieval strategy
based on complexity analysis. Implements Chain-of-Thought decomposition
for complex multi-hop queries.
Based on: Adaptive-RAG (NAACL 2024, arXiv:2403.14403)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import RetrievalStrategy, settings

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = (
    Path(__file__).parent.parent / "config" / "prompts" / "query_decomposition.txt"
).read_text()


class QueryType(str, Enum):
    FACTOID = "factoid"  # Simple fact lookup
    COMPARISON = "comparison"  # Comparing entities
    CAUSAL = "causal"  # Why/how questions
    TEMPORAL = "temporal"  # Time-based queries
    AGGREGATION = "aggregation"  # Summarize/list all
    MULTI_HOP = "multi_hop"  # Requires chaining multiple facts


@dataclass
class SubQuestion:
    id: int
    question: str
    purpose: str
    depends_on: list[int] = field(default_factory=list)
    answer: Optional[str] = None


@dataclass
class RetrievalPlan:
    strategy: RetrievalStrategy
    query_type: QueryType
    original_query: str
    sub_questions: list[SubQuestion] = field(default_factory=list)
    target_domains: list[int] = field(default_factory=list)  # Community IDs
    hyde_enabled: bool = True
    confidence: float = 0.0
    reasoning: str = ""

    def is_multi_hop(self) -> bool:
        return self.strategy == RetrievalStrategy.COMPLEX and len(self.sub_questions) > 1


class QueryRouter:
    """Classifies query complexity and builds a retrieval plan."""

    # Complexity signals
    COMPARISON_WORDS = {
        "compare",
        "versus",
        "vs",
        "difference",
        "differ",
        "contrast",
        "better",
        "worse",
        "advantage",
        "disadvantage",
        "pros",
        "cons",
    }
    CAUSAL_WORDS = {
        "why",
        "cause",
        "because",
        "reason",
        "result",
        "effect",
        "impact",
        "consequence",
        "lead to",
        "due to",
    }
    TEMPORAL_WORDS = {
        "when",
        "before",
        "after",
        "during",
        "timeline",
        "history",
        "first",
        "last",
        "recent",
        "earliest",
        "latest",
        "evolution",
    }
    AGGREGATION_WORDS = {
        "all",
        "every",
        "list",
        "summarize",
        "overview",
        "main themes",
        "what are the",
        "how many",
        "enumerate",
    }
    MULTI_HOP_SIGNALS = {
        "and also",
        "in addition",
        "furthermore",
        "as well as",
        "both",
        "relationship between",
        "how does.*relate",
        "connection between",
    }

    def __init__(self, model: str = settings.fast_model):
        self.model = model

    def _classify_query_type(self, query: str) -> QueryType:
        """Rule-based query type classification."""
        query_lower = query.lower()

        if any(w in query_lower for w in self.COMPARISON_WORDS):
            return QueryType.COMPARISON
        if any(w in query_lower for w in self.CAUSAL_WORDS):
            return QueryType.CAUSAL
        if any(w in query_lower for w in self.TEMPORAL_WORDS):
            return QueryType.TEMPORAL
        if any(w in query_lower for w in self.AGGREGATION_WORDS):
            return QueryType.AGGREGATION
        if any(re.search(pattern, query_lower) for pattern in self.MULTI_HOP_SIGNALS):
            return QueryType.MULTI_HOP
        return QueryType.FACTOID

    def _estimate_complexity(self, query: str) -> tuple[RetrievalStrategy, float, str]:
        """Estimate query complexity using heuristic signals."""
        query_lower = query.lower()
        score = 0.0
        reasons = []

        # Word count signal
        word_count = len(query.split())
        if word_count > 25:
            score += 0.3
            reasons.append(f"long query ({word_count} words)")
        elif word_count > 12:
            score += 0.15
            reasons.append(f"medium query ({word_count} words)")

        # Question mark count (compound questions)
        q_marks = query.count("?")
        if q_marks > 1:
            score += 0.25
            reasons.append(f"compound question ({q_marks} question marks)")

        # Entity count (approximated by capitalized words)
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", query)
        if len(entities) > 3:
            score += 0.2
            reasons.append(f"many entities ({len(entities)})")
        elif len(entities) > 1:
            score += 0.1

        # Comparison/multi-hop signals
        if any(w in query_lower for w in self.COMPARISON_WORDS):
            score += 0.2
            reasons.append("comparison query")
        if any(re.search(p, query_lower) for p in self.MULTI_HOP_SIGNALS):
            score += 0.3
            reasons.append("multi-hop signals detected")

        # Causal reasoning
        if any(w in query_lower for w in self.CAUSAL_WORDS):
            score += 0.15
            reasons.append("causal reasoning needed")

        # Aggregation
        if any(w in query_lower for w in self.AGGREGATION_WORDS):
            score += 0.2
            reasons.append("aggregation query")

        # Determine strategy
        if score < 0.2:
            strategy = RetrievalStrategy.SIMPLE
        elif score < 0.5:
            strategy = RetrievalStrategy.MODERATE
        else:
            strategy = RetrievalStrategy.COMPLEX

        reasoning = "; ".join(reasons) if reasons else "simple factoid query"
        return strategy, min(score, 1.0), reasoning

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _decompose_query(self, query: str) -> list[SubQuestion]:
        """Use LLM to decompose a complex query into sub-questions (CoT)."""
        prompt = DECOMPOSE_PROMPT.replace("{query}", query)

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a query decomposition expert. Return ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        sub_questions = []
        for sq in result.get("sub_questions", []):
            sub_questions.append(
                SubQuestion(
                    id=sq.get("id", len(sub_questions) + 1),
                    question=sq.get("question", ""),
                    purpose=sq.get("purpose", ""),
                    depends_on=sq.get("depends_on", []),
                )
            )
        return sub_questions

    async def route(self, query: str) -> RetrievalPlan:
        """Analyze a query and produce a retrieval plan."""
        query_type = self._classify_query_type(query)
        strategy, confidence, reasoning = self._estimate_complexity(query)

        plan = RetrievalPlan(
            strategy=strategy,
            query_type=query_type,
            original_query=query,
            hyde_enabled=settings.hyde_enabled and strategy != RetrievalStrategy.SIMPLE,
            confidence=confidence,
            reasoning=reasoning,
        )

        # For complex queries, decompose into sub-questions
        if strategy == RetrievalStrategy.COMPLEX:
            try:
                plan.sub_questions = await self._decompose_query(query)
                logger.info(
                    f"Decomposed complex query into {len(plan.sub_questions)} sub-questions"
                )
            except Exception as e:
                logger.warning(f"Query decomposition failed, treating as moderate: {e}")
                plan.strategy = RetrievalStrategy.MODERATE

        logger.info(
            f"Routed query: strategy={plan.strategy.value}, "
            f"type={plan.query_type.value}, confidence={plan.confidence:.2f}, "
            f"sub_questions={len(plan.sub_questions)}"
        )
        return plan

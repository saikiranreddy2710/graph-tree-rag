"""
Self-RAG Reflection — Self-reflective verification of generated answers.
Evaluates relevance, support, usefulness, and citation accuracy.
Triggers corrective loops when quality is insufficient.
Based on: Self-RAG (arXiv:2310.11511)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from retrieval.hybrid_fuser import FusedResult

logger = logging.getLogger(__name__)

REFLECTION_PROMPT = (
    Path(__file__).parent.parent / "config" / "prompts" / "self_rag_reflection.txt"
).read_text()


@dataclass
class ReflectionScore:
    relevance: float = 0.0
    support: float = 0.0
    usefulness: float = 0.0
    citations: float = 0.0
    overall_pass: bool = False
    recommended_action: str = "ACCEPT"
    unsupported_claims: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    reasoning: dict[str, str] = field(default_factory=dict)

    @property
    def average_score(self) -> float:
        return (self.relevance + self.support + self.usefulness + self.citations) / 4


@dataclass
class SelfRAGResult:
    final_answer: str
    reflection: ReflectionScore
    iterations: int = 0
    answer_history: list[str] = field(default_factory=list)
    grounded_answer: str = ""  # Answer with verified citations


class SelfRAGReflector:
    """Reflects on generated answers and triggers corrective loops."""

    def __init__(
        self,
        model: str = settings.fast_model,
        max_loops: int = settings.self_rag_max_loops,
        support_threshold: float = settings.self_rag_support_threshold,
    ):
        self.model = model
        self.max_loops = max_loops
        self.support_threshold = support_threshold

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _reflect(
        self, query: str, answer: str, context: list[FusedResult]
    ) -> ReflectionScore:
        """Run reflection evaluation on a generated answer."""
        context_text = "\n\n".join(f"[{r.doc_id}]: {r.text[:400]}" for r in context[:8])

        prompt = (
            REFLECTION_PROMPT.replace("{query}", query)
            .replace("{answer}", answer)
            .replace("{context}", context_text)
        )

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a critical evaluator. Return ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=768,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        return ReflectionScore(
            relevance=float(result.get("relevance", {}).get("score", 0.5)),
            support=float(result.get("support", {}).get("score", 0.5)),
            usefulness=float(result.get("usefulness", {}).get("score", 0.5)),
            citations=float(result.get("citations", {}).get("score", 0.5)),
            overall_pass=result.get("overall_pass", False),
            recommended_action=result.get("recommended_action", "ACCEPT"),
            unsupported_claims=result.get("support", {}).get("unsupported_claims", []),
            missing_citations=result.get("citations", {}).get("missing_citations", []),
            reasoning={
                "relevance": result.get("relevance", {}).get("explanation", ""),
                "support": result.get("support", {}).get("explanation", ""),
                "usefulness": result.get("usefulness", {}).get("explanation", ""),
                "citations": result.get("citations", {}).get("explanation", ""),
            },
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _regenerate_with_feedback(
        self,
        query: str,
        previous_answer: str,
        reflection: ReflectionScore,
        context: list[FusedResult],
    ) -> str:
        """Regenerate answer incorporating reflection feedback."""
        context_text = "\n\n".join(f"[{r.doc_id}]: {r.text[:400]}" for r in context[:8])

        feedback_parts = []
        if reflection.unsupported_claims:
            feedback_parts.append(
                f"Remove or fix these unsupported claims: {reflection.unsupported_claims}"
            )
        if reflection.missing_citations:
            feedback_parts.append(f"Add citations for: {reflection.missing_citations}")
        if reflection.support < self.support_threshold:
            feedback_parts.append("Ensure every factual claim is directly supported by the context")

        feedback = "\n".join(feedback_parts)

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise question-answering assistant. "
                        "Regenerate the answer addressing the feedback below. "
                        "Use ONLY information from the context. "
                        "Add [doc_id] citations for every factual claim."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"## Question:\n{query}\n\n"
                        f"## Context:\n{context_text}\n\n"
                        f"## Previous Answer (needs improvement):\n{previous_answer}\n\n"
                        f"## Feedback to address:\n{feedback}\n\n"
                        f"## Improved Answer:"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    async def reflect_and_correct(
        self,
        query: str,
        answer: str,
        context: list[FusedResult],
    ) -> SelfRAGResult:
        """
        Full Self-RAG loop:
        1. Reflect on the answer
        2. If quality is insufficient, regenerate with feedback
        3. Repeat up to max_loops times
        """
        current_answer = answer
        answer_history = [answer]
        iterations = 0

        for loop in range(self.max_loops + 1):
            # Reflect
            reflection = await self._reflect(query, current_answer, context)
            iterations = loop + 1

            logger.info(
                f"Self-RAG loop {loop}: "
                f"relevance={reflection.relevance:.2f}, "
                f"support={reflection.support:.2f}, "
                f"usefulness={reflection.usefulness:.2f}, "
                f"citations={reflection.citations:.2f}, "
                f"action={reflection.recommended_action}"
            )

            # Check if answer is acceptable
            if reflection.overall_pass or reflection.recommended_action == "ACCEPT":
                break

            # If we've hit the max loops, accept what we have
            if loop >= self.max_loops:
                break

            # Regenerate with feedback
            try:
                current_answer = await self._regenerate_with_feedback(
                    query, current_answer, reflection, context
                )
                answer_history.append(current_answer)
            except Exception as e:
                logger.error(f"Self-RAG regeneration failed: {e}")
                break

        return SelfRAGResult(
            final_answer=current_answer,
            reflection=reflection,
            iterations=iterations,
            answer_history=answer_history,
            grounded_answer=current_answer,
        )

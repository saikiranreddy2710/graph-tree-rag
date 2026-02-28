"""
CRAG Evaluator — Corrective Retrieval-Augmented Generation.
Evaluates retrieval quality and triggers corrective actions:
  CORRECT  -> proceed to generation
  AMBIGUOUS -> decompose & re-retrieve
  INCORRECT -> web search fallback
Based on: CRAG (arXiv:2401.15884)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from retrieval.hybrid_fuser import FusedResult

logger = logging.getLogger(__name__)

CRAG_PROMPT = (
    Path(__file__).parent.parent / "config" / "prompts" / "crag_evaluation.txt"
).read_text()


class RetrievalQuality(str, Enum):
    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"


@dataclass
class CRAGEvaluation:
    quality: RetrievalQuality
    confidence: float
    reasoning: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    irrelevant_doc_ids: list[str] = field(default_factory=list)
    missing_information: str = ""
    recommended_action: str = "PROCEED"
    filtered_results: list[FusedResult] = field(default_factory=list)
    web_results: list[dict] = field(default_factory=list)


class CRAGEvaluator:
    """Evaluates retrieval quality and triggers corrective actions."""

    def __init__(
        self,
        model: str = settings.fast_model,
        confidence_threshold: float = settings.crag_confidence_threshold,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _evaluate_retrieval(self, query: str, results: list[FusedResult]) -> dict:
        """LLM-based evaluation of retrieval quality."""
        docs_text = "\n\n".join(
            f"[DOC {i + 1} | id={r.doc_id}]\n{r.text[:500]}" for i, r in enumerate(results[:10])
        )

        prompt = CRAG_PROMPT.replace("{query}", query).replace("{documents}", docs_text)

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a retrieval quality evaluator. Return ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    async def _web_search_fallback(self, query: str) -> list[dict]:
        """Fallback to web search when retrieval is insufficient."""
        if not settings.web_search_enabled or not settings.web_search_api_key:
            logger.warning("Web search fallback requested but not configured")
            return []

        try:
            import httpx

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": settings.web_search_api_key,
                        "query": query,
                        "max_results": settings.web_search_max_results,
                        "include_raw_content": False,
                    },
                )
                response.raise_for_status()
                data = response.json()

                web_results = []
                for result in data.get("results", []):
                    web_results.append(
                        {
                            "title": result.get("title", ""),
                            "text": result.get("content", ""),
                            "url": result.get("url", ""),
                            "score": result.get("score", 0.0),
                            "source_type": "web_search",
                        }
                    )

                logger.info(f"Web search returned {len(web_results)} results")
                return web_results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def _filter_relevant_results(
        self,
        results: list[FusedResult],
        relevant_ids: list[str],
        irrelevant_ids: list[str],
    ) -> list[FusedResult]:
        """Keep only documents the evaluator deemed relevant."""
        if not relevant_ids and not irrelevant_ids:
            return results

        irrelevant_set = set(irrelevant_ids)
        return [r for r in results if r.doc_id not in irrelevant_set]

    def _extract_key_knowledge(self, results: list[FusedResult]) -> list[FusedResult]:
        """
        Decompose-then-recompose: strip irrelevant portions from passages.
        For now, we keep passages but could add sentence-level filtering.
        """
        # Simple heuristic: trim very long passages to their most relevant portion
        filtered = []
        for r in results:
            if len(r.text) > 2000:
                # Keep first 2000 chars as approximation
                # A more advanced version would use sentence-level relevance
                r_copy = FusedResult(
                    doc_id=r.doc_id,
                    text=r.text[:2000] + "...",
                    fused_score=r.fused_score,
                    channel_scores=r.channel_scores,
                    channel_ranks=r.channel_ranks,
                    source_type=r.source_type,
                    metadata=r.metadata,
                    entities_matched=r.entities_matched,
                )
                filtered.append(r_copy)
            else:
                filtered.append(r)
        return filtered

    async def evaluate(self, query: str, results: list[FusedResult]) -> CRAGEvaluation:
        """
        Full CRAG evaluation pipeline:
        1. Assess retrieval quality
        2. Filter irrelevant docs
        3. If INCORRECT -> web search fallback
        4. If AMBIGUOUS -> extract key knowledge, flag for decomposition
        """
        if not results:
            return CRAGEvaluation(
                quality=RetrievalQuality.INCORRECT,
                confidence=0.0,
                reasoning="No retrieval results",
                recommended_action="WEB_SEARCH_FALLBACK",
            )

        # Step 1: LLM evaluation
        try:
            eval_result = await self._evaluate_retrieval(query, results)
        except Exception as e:
            logger.error(f"CRAG evaluation failed: {e}")
            # Default to AMBIGUOUS if evaluation fails
            return CRAGEvaluation(
                quality=RetrievalQuality.AMBIGUOUS,
                confidence=0.5,
                reasoning=f"Evaluation failed: {e}",
                filtered_results=results,
                recommended_action="PROCEED",
            )

        quality = RetrievalQuality(eval_result.get("overall_quality", "AMBIGUOUS"))
        confidence = float(eval_result.get("confidence", 0.5))
        relevant_ids = eval_result.get("relevant_doc_ids", [])
        irrelevant_ids = eval_result.get("irrelevant_doc_ids", [])

        evaluation = CRAGEvaluation(
            quality=quality,
            confidence=confidence,
            reasoning=eval_result.get("reasoning", ""),
            relevant_doc_ids=relevant_ids,
            irrelevant_doc_ids=irrelevant_ids,
            missing_information=eval_result.get("missing_information", ""),
            recommended_action=eval_result.get("recommended_action", "PROCEED"),
        )

        # Step 2: Filter results
        filtered = self._filter_relevant_results(results, relevant_ids, irrelevant_ids)
        filtered = self._extract_key_knowledge(filtered)
        evaluation.filtered_results = filtered

        # Step 3: Corrective actions
        if quality == RetrievalQuality.INCORRECT:
            logger.warning("CRAG: Retrieval quality INCORRECT, triggering web search")
            web_results = await self._web_search_fallback(query)
            evaluation.web_results = web_results
            evaluation.recommended_action = "WEB_SEARCH_FALLBACK"

        elif quality == RetrievalQuality.AMBIGUOUS:
            logger.info("CRAG: Retrieval quality AMBIGUOUS, recommending decomposition")
            evaluation.recommended_action = "DECOMPOSE_AND_RETRY"

        else:
            logger.info(f"CRAG: Retrieval quality CORRECT (confidence={confidence:.2f})")
            evaluation.recommended_action = "PROCEED"

        return evaluation

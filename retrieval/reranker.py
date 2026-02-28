"""
Re-ranker — Cross-encoder re-ranking with graph-coherence bonus and MMR diversity.
Applies fine-grained relevance scoring, graph-coherence bonuses, temporal recency,
and Maximal Marginal Relevance for deduplication.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np

from config.settings import settings
from ingestion.graph_builder import KnowledgeGraph
from retrieval.hybrid_fuser import FusedResult

logger = logging.getLogger(__name__)


class ReRanker:
    """Multi-signal re-ranker with cross-encoder, graph-coherence, and MMR."""

    def __init__(
        self,
        kg: Optional[KnowledgeGraph] = None,
        model_name: str = settings.reranker_model,
        mmr_lambda: float = settings.mmr_lambda,
        graph_coherence_bonus: float = settings.graph_coherence_bonus,
        temporal_bonus: float = settings.temporal_recency_bonus,
    ):
        self.kg = kg
        self.model_name = model_name
        self.mmr_lambda = mmr_lambda
        self.graph_coherence_bonus = graph_coherence_bonus
        self.temporal_bonus = temporal_bonus
        self._cross_encoder = None
        self._embedder = None

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.model_name)
        return self._cross_encoder

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(settings.embedding_model)
        return self._embedder

    def _cross_encoder_scores(self, query: str, results: list[FusedResult]) -> list[float]:
        """Score each (query, passage) pair with a cross-encoder."""
        if not results:
            return []

        ce = self._get_cross_encoder()
        pairs = [(query, r.text[:512]) for r in results]  # Truncate for efficiency
        scores = ce.predict(pairs)
        return [float(s) for s in scores]

    def _graph_coherence_scores(
        self, query_entities: list[str], results: list[FusedResult]
    ) -> list[float]:
        """Score each result by how many of its entities are connected to query entities."""
        if not self.kg or not query_entities:
            return [0.0] * len(results)

        scores = []
        query_entity_set = set(e.lower() for e in query_entities)

        for result in results:
            bonus = 0.0
            result_entities = [e.lower() for e in result.entities_matched]

            for re_ent in result_entities:
                if re_ent in query_entity_set:
                    bonus += self.graph_coherence_bonus
                    continue

                # Check if connected in KG
                for qe in query_entity_set:
                    # Check direct connection
                    for node in self.kg.graph.nodes:
                        if node.lower() == re_ent:
                            for _, tgt, _ in self.kg.graph.out_edges(node, data=True):
                                if tgt.lower() in query_entity_set:
                                    bonus += self.graph_coherence_bonus * 0.5
                            break

            scores.append(min(bonus, self.graph_coherence_bonus * 3))

        return scores

    def _temporal_scores(self, results: list[FusedResult]) -> list[float]:
        """Compute temporal recency bonus based on document creation date."""
        scores = []
        now = datetime.utcnow()

        for result in results:
            created_at = result.metadata.get("created_at")
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at)
                    days_old = (now - dt).days
                    # Exponential decay: half-life of 365 days
                    recency = self.temporal_bonus * (0.5 ** (days_old / 365))
                    scores.append(recency)
                except (ValueError, TypeError):
                    scores.append(0.0)
            else:
                scores.append(0.0)

        return scores

    def _mmr_diversify(
        self,
        results: list[FusedResult],
        scores: list[float],
        top_k: int,
    ) -> list[tuple[FusedResult, float]]:
        """
        Maximal Marginal Relevance for diversity-aware selection.
        Balances relevance (score) with diversity (dissimilarity to already selected).
        """
        if len(results) <= top_k:
            return list(zip(results, scores))

        # Embed all result texts for similarity computation
        embedder = self._get_embedder()
        texts = [r.text[:512] for r in results]
        embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        selected_indices: list[int] = []
        remaining_indices = list(range(len(results)))

        for _ in range(top_k):
            if not remaining_indices:
                break

            best_idx = None
            best_mmr = -float("inf")

            for idx in remaining_indices:
                relevance = scores[idx]

                # Max similarity to already selected documents
                max_sim = 0.0
                if selected_indices:
                    for sel_idx in selected_indices:
                        sim = float(np.dot(embeddings[idx], embeddings[sel_idx]))
                        max_sim = max(max_sim, sim)

                # MMR = lambda * relevance - (1 - lambda) * max_similarity
                mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        return [(results[i], scores[i]) for i in selected_indices]

    def rerank(
        self,
        query: str,
        results: list[FusedResult],
        query_entities: Optional[list[str]] = None,
        top_k: int = settings.reranker_top_k,
    ) -> list[FusedResult]:
        """
        Full re-ranking pipeline:
        1. Cross-encoder fine-grained scoring
        2. Graph-coherence bonus
        3. Temporal recency bonus
        4. MMR diversity-aware selection
        """
        if not results:
            return []

        # Step 1: Cross-encoder scores
        ce_scores = self._cross_encoder_scores(query, results)

        # Step 2: Graph-coherence bonus
        gc_scores = self._graph_coherence_scores(query_entities or [], results)

        # Step 3: Temporal bonus
        temp_scores = self._temporal_scores(results)

        # Combine scores
        combined_scores = []
        for i in range(len(results)):
            combined = ce_scores[i] + gc_scores[i] + temp_scores[i]
            combined_scores.append(combined)

        # Step 4: MMR diversification
        selected = self._mmr_diversify(results, combined_scores, top_k)

        # Update fused scores and return
        reranked = []
        for result, score in selected:
            result.fused_score = score
            result.metadata["ce_score"] = ce_scores[results.index(result)]
            result.metadata["gc_bonus"] = gc_scores[results.index(result)]
            result.metadata["temporal_bonus"] = temp_scores[results.index(result)]
            reranked.append(result)

        logger.info(
            f"Re-ranked {len(results)} -> {len(reranked)} results "
            f"(CE range: {min(ce_scores):.3f}-{max(ce_scores):.3f})"
        )
        return reranked

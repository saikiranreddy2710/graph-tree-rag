"""
Hybrid Fusion Engine — Reciprocal Rank Fusion (RRF) across 4 retrieval channels.
Merges results from vector search, BM25, graph traversal, and tree retrieval
using weighted RRF with configurable per-channel weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class FusedResult:
    """A unified retrieval result after fusion across channels."""

    doc_id: str  # chunk_id, node_id, or hash of text
    text: str
    fused_score: float
    channel_scores: dict[str, float] = field(default_factory=dict)
    channel_ranks: dict[str, int] = field(default_factory=dict)
    source_type: str = "chunk"
    metadata: dict[str, Any] = field(default_factory=dict)
    entities_matched: list[str] = field(default_factory=list)


class HybridFuser:
    """Fuses results from multiple retrieval channels using weighted RRF."""

    def __init__(
        self,
        weight_vector: float = settings.fusion_weights_vector,
        weight_bm25: float = settings.fusion_weights_bm25,
        weight_graph: float = settings.fusion_weights_graph,
        weight_tree: float = settings.fusion_weights_tree,
        rrf_k: int = settings.fusion_rrf_k,
    ):
        self.weights = {
            "vector": weight_vector,
            "bm25": weight_bm25,
            "graph": weight_graph,
            "tree": weight_tree,
        }
        self.rrf_k = rrf_k

    def _extract_doc_id(self, result: dict, channel: str) -> str:
        """Extract a consistent document identifier across channels."""
        if "chunk_id" in result and result["chunk_id"]:
            return result["chunk_id"]
        if "node_id" in result and result["node_id"]:
            return result["node_id"]
        if "bm25_id" in result and result["bm25_id"]:
            return result["bm25_id"]
        if "doc_id" in result:
            return result["doc_id"]
        # Fallback: hash of text
        text = result.get("text", "")
        return f"{channel}_{hash(text[:200])}"

    def _extract_text(self, result: dict) -> str:
        return result.get("text", "")

    def fuse(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        graph_results: list[dict],
        tree_results: list[dict],
        top_k: int = settings.final_top_k,
    ) -> list[FusedResult]:
        """
        Fuse results from all channels using Reciprocal Rank Fusion.

        RRF score for document d:
            score(d) = sum( w_c / (k + rank_c(d)) ) for each channel c

        This is robust to score normalization differences across channels.
        """
        channels = {
            "vector": vector_results,
            "bm25": bm25_results,
            "graph": graph_results,
            "tree": tree_results,
        }

        # Build per-document aggregated scores
        doc_scores: dict[str, FusedResult] = {}

        for channel_name, results in channels.items():
            weight = self.weights.get(channel_name, 0.25)

            for rank, result in enumerate(results, start=1):
                doc_id = self._extract_doc_id(result, channel_name)
                text = self._extract_text(result)
                raw_score = result.get("score", 0.0)

                # RRF contribution
                rrf_contribution = weight / (self.rrf_k + rank)

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = FusedResult(
                        doc_id=doc_id,
                        text=text,
                        fused_score=0.0,
                        source_type=result.get("type", result.get("source_type", "chunk")),
                        metadata=result.get("metadata", {}),
                        entities_matched=result.get("entities_matched", []),
                    )

                fused = doc_scores[doc_id]
                fused.fused_score += rrf_contribution
                fused.channel_scores[channel_name] = raw_score
                fused.channel_ranks[channel_name] = rank

                # Merge entities
                new_entities = result.get("entities_matched", [])
                if new_entities:
                    fused.entities_matched = list(set(fused.entities_matched + new_entities))

                # Keep the longest text version
                if len(text) > len(fused.text):
                    fused.text = text

        # Sort by fused score
        ranked = sorted(
            doc_scores.values(),
            key=lambda x: x.fused_score,
            reverse=True,
        )

        # Log channel contribution stats
        if ranked:
            channel_hit_counts = {ch: 0 for ch in channels}
            for r in ranked[:top_k]:
                for ch in r.channel_scores:
                    channel_hit_counts[ch] += 1
            logger.info(
                f"Fusion: {len(ranked)} unique docs, top-{top_k} channel hits: {channel_hit_counts}"
            )

        return ranked[:top_k]

"""
RAPTOR Tree Retriever — Collapsed-tree search across all abstraction levels.
Retrieves from leaf (specific) and internal (abstract) nodes simultaneously.
Based on: RAPTOR (arXiv:2401.18059)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from config.settings import settings
from ingestion.tree_builder import RaptorTree

logger = logging.getLogger(__name__)


@dataclass
class TreeRetrievalResult:
    node_id: str
    text: str
    score: float
    level: int
    source_type: str = "tree_node"
    metadata: dict = field(default_factory=dict)


class TreeRetriever:
    """Retrieves from the RAPTOR tree using collapsed-tree search."""

    def __init__(self, tree: RaptorTree):
        self.tree = tree
        self._embedder = None
        self._node_embeddings: dict[str, np.ndarray] = {}
        self._build_embedding_cache()

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(settings.embedding_model)
        return self._embedder

    def _build_embedding_cache(self) -> None:
        """Cache all node embeddings as numpy arrays for fast retrieval."""
        for node_id, node in self.tree.nodes.items():
            if node.embedding:
                self._node_embeddings[node_id] = np.array(node.embedding, dtype=np.float32)

        # For nodes without pre-computed embeddings, compute them
        missing = [
            (nid, node) for nid, node in self.tree.nodes.items() if nid not in self._node_embeddings
        ]
        if missing:
            embedder = self._get_embedder()
            texts = [node.text for _, node in missing]
            embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            for (nid, _), emb in zip(missing, embeddings):
                self._node_embeddings[nid] = emb.astype(np.float32)

    def _compute_level_weight(self, level: int) -> float:
        """
        Compute a weight multiplier for a tree level.
        Lower levels (leaves) get higher specificity bonus.
        Higher levels (summaries) get broader context bonus.
        """
        max_level = max(self.tree.max_level, 1)
        # Slight preference for mid-level nodes that balance specificity and context
        mid = max_level / 2
        distance_from_mid = abs(level - mid) / max_level
        return 1.0 - 0.1 * distance_from_mid

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = settings.tree_top_k,
    ) -> list[TreeRetrievalResult]:
        """
        Collapsed-tree search: search across ALL tree levels simultaneously.
        This is the key RAPTOR insight -- retrieve from any abstraction level.
        """
        if not self._node_embeddings:
            return []

        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]
        query_embedding = query_embedding.astype(np.float32)

        # Compute similarity against all nodes
        scores = []
        for node_id, node_emb in self._node_embeddings.items():
            similarity = float(np.dot(query_embedding, node_emb))
            node = self.tree.nodes[node_id]
            level_weight = self._compute_level_weight(node.level)
            weighted_score = similarity * level_weight
            scores.append((node_id, weighted_score, similarity))

        # Sort by weighted score
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for node_id, weighted_score, raw_score in scores[:top_k]:
            node = self.tree.nodes[node_id]
            results.append(
                TreeRetrievalResult(
                    node_id=node_id,
                    text=node.text,
                    score=weighted_score,
                    level=node.level,
                    metadata={
                        "raw_similarity": raw_score,
                        "source_chunk_id": node.source_chunk_id,
                        "children_count": len(node.children),
                        "is_leaf": node.level == 0,
                    },
                )
            )

        logger.info(
            f"Tree retrieval: found {len(results)} nodes across levels "
            f"{set(r.level for r in results)}"
        )
        return results

"""
RAPTOR Tree Builder — Recursive Abstractive Processing for Tree-Organized Retrieval.
Builds a bottom-up summarization tree via embed -> cluster -> summarize -> recurse.
Based on: arXiv:2401.18059 (RAPTOR, Stanford)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import litellm
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from ingestion.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    node_id: str
    text: str
    level: int  # 0 = leaf (raw chunk), 1+ = summary nodes
    children: list[str] = field(default_factory=list)  # child node_ids
    parent: Optional[str] = None
    embedding: Optional[list[float]] = None
    source_chunk_id: Optional[str] = None  # Only for leaf nodes
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "level": self.level,
            "children": self.children,
            "parent": self.parent,
            "embedding": self.embedding,
            "source_chunk_id": self.source_chunk_id,
            "metadata": self.metadata,
        }


@dataclass
class RaptorTree:
    nodes: dict[str, TreeNode] = field(default_factory=dict)
    root_ids: list[str] = field(default_factory=list)
    max_level: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_nodes_at_level(self, level: int) -> list[TreeNode]:
        return [n for n in self.nodes.values() if n.level == level]

    def get_all_nodes_flat(self) -> list[TreeNode]:
        """Collapsed tree: return all nodes across all levels for retrieval."""
        return list(self.nodes.values())

    def get_leaf_nodes(self) -> list[TreeNode]:
        return self.get_nodes_at_level(0)

    def to_dict(self) -> dict:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "root_ids": self.root_ids,
            "max_level": self.max_level,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RaptorTree:
        tree = cls()
        for k, v in data.get("nodes", {}).items():
            tree.nodes[k] = TreeNode(**v)
        tree.root_ids = data.get("root_ids", [])
        tree.max_level = data.get("max_level", 0)
        tree.metadata = data.get("metadata", {})
        return tree


class TreeBuilder:
    """Builds a RAPTOR-style hierarchical summarization tree."""

    def __init__(
        self,
        model: str = settings.fast_model,
        max_levels: int = settings.tree_max_levels,
        cluster_dim: int = settings.tree_cluster_dim,
        min_cluster_size: int = settings.tree_min_cluster_size,
    ):
        self.model = model
        self.max_levels = max_levels
        self.cluster_dim = cluster_dim
        self.min_cluster_size = min_cluster_size
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(settings.embedding_model)
        return self._embedder

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        embedder = self._get_embedder()
        return embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def _cluster_embeddings(
        self, embeddings: np.ndarray, n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """Cluster embeddings using UMAP reduction + GMM."""
        from sklearn.mixture import GaussianMixture

        n_samples = len(embeddings)
        if n_samples < self.min_cluster_size:
            return np.zeros(n_samples, dtype=int)

        # Determine optimal number of clusters
        if n_clusters is None:
            n_clusters = max(2, min(n_samples // self.min_cluster_size, 20))

        # Dimensionality reduction with UMAP if embeddings are high-dimensional
        reduced = embeddings
        if embeddings.shape[1] > self.cluster_dim and n_samples > self.cluster_dim + 2:
            try:
                from umap import UMAP

                reducer = UMAP(
                    n_components=min(self.cluster_dim, n_samples - 2),
                    metric="cosine",
                    random_state=42,
                )
                reduced = reducer.fit_transform(embeddings)
            except Exception as e:
                logger.warning(f"UMAP reduction failed, using raw embeddings: {e}")

        # Gaussian Mixture Model clustering
        n_clusters = min(n_clusters, n_samples)
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=42,
        )
        labels = gmm.fit_predict(reduced)
        return labels

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def _summarize_cluster(self, texts: list[str]) -> str:
        """Generate an abstractive summary for a cluster of texts."""
        combined = "\n\n---\n\n".join(texts[:10])  # Limit for context window

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise summarization engine. Synthesize the following "
                        "text passages into a single coherent summary that captures ALL key "
                        "information, entities, facts, and relationships. The summary should "
                        "be 2-4 paragraphs and self-contained."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Summarize these related passages:\n\n{combined}",
                },
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    async def build(self, chunks: list[DocumentChunk]) -> RaptorTree:
        """Build the complete RAPTOR tree from document chunks."""
        tree = RaptorTree()
        tree.metadata = {
            "num_leaves": len(chunks),
            "max_levels": self.max_levels,
        }

        if not chunks:
            return tree

        # Level 0: Create leaf nodes from chunks
        leaf_nodes = []
        for chunk in chunks:
            node = TreeNode(
                node_id=f"leaf_{chunk.chunk_id}",
                text=chunk.text,
                level=0,
                source_chunk_id=chunk.chunk_id,
                metadata={
                    "source": chunk.metadata.source,
                    "section": chunk.metadata.section_header,
                },
            )
            tree.nodes[node.node_id] = node
            leaf_nodes.append(node)

        # Embed all leaf nodes
        leaf_texts = [n.text for n in leaf_nodes]
        leaf_embeddings = self._embed_texts(leaf_texts)
        for node, emb in zip(leaf_nodes, leaf_embeddings):
            node.embedding = emb.tolist()

        # Build tree bottom-up
        current_level_nodes = leaf_nodes
        current_level_embeddings = leaf_embeddings

        for level in range(1, self.max_levels + 1):
            if len(current_level_nodes) <= 1:
                break

            logger.info(f"Building tree level {level} from {len(current_level_nodes)} nodes")

            # Cluster current level
            labels = self._cluster_embeddings(current_level_embeddings)
            unique_labels = set(labels)

            # Create summary nodes for each cluster
            next_level_nodes = []
            for label in unique_labels:
                cluster_indices = [i for i, l in enumerate(labels) if l == label]
                if len(cluster_indices) < 1:
                    continue

                cluster_nodes = [current_level_nodes[i] for i in cluster_indices]
                cluster_texts = [n.text for n in cluster_nodes]

                # Generate summary
                summary_text = await self._summarize_cluster(cluster_texts)

                # Create parent node
                parent_id = f"L{level}_{uuid.uuid4().hex[:8]}"
                parent_node = TreeNode(
                    node_id=parent_id,
                    text=summary_text,
                    level=level,
                    children=[n.node_id for n in cluster_nodes],
                )

                # Update children to point to parent
                for child in cluster_nodes:
                    child.parent = parent_id

                tree.nodes[parent_id] = parent_node
                next_level_nodes.append(parent_node)

            if not next_level_nodes:
                break

            # Embed the new summary nodes
            summary_texts = [n.text for n in next_level_nodes]
            summary_embeddings = self._embed_texts(summary_texts)
            for node, emb in zip(next_level_nodes, summary_embeddings):
                node.embedding = emb.tolist()

            current_level_nodes = next_level_nodes
            current_level_embeddings = summary_embeddings
            tree.max_level = level

        # Set root nodes
        tree.root_ids = [n.node_id for n in current_level_nodes]

        logger.info(
            f"Built RAPTOR tree: {len(tree.nodes)} total nodes, "
            f"{tree.max_level + 1} levels, {len(tree.root_ids)} roots"
        )
        return tree

    def save(self, tree: RaptorTree, path: Optional[Path] = None) -> None:
        path = path or settings.tree_persist_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(tree.to_dict(), f, indent=2)
        logger.info(f"Saved RAPTOR tree to {path}")

    def load(self, path: Optional[Path] = None) -> RaptorTree:
        path = path or settings.tree_persist_path
        with open(path) as f:
            data = json.load(f)
        tree = RaptorTree.from_dict(data)
        logger.info(f"Loaded RAPTOR tree: {len(tree.nodes)} nodes")
        return tree

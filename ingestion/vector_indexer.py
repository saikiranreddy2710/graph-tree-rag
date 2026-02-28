"""
Vector Indexer — FAISS-based dense vector index + BM25 sparse index.
Supports both document chunks and RAPTOR tree nodes.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import settings
from ingestion.document_processor import DocumentChunk
from ingestion.tree_builder import RaptorTree, TreeNode

logger = logging.getLogger(__name__)


class VectorIndexer:
    """Manages FAISS dense index and BM25 sparse index."""

    def __init__(self, embedding_dim: int = settings.embedding_dimension):
        self.embedding_dim = embedding_dim
        self._embedder = None

        # Dense index
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.id_to_metadata: dict[int, dict] = {}

        # Sparse index
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: list[str] = []
        self.bm25_ids: list[str] = []

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(settings.embedding_model)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        embedder = self._get_embedder()
        embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def build_from_chunks(
        self,
        chunks: list[DocumentChunk],
        tree: Optional[RaptorTree] = None,
    ) -> None:
        """Build both dense and sparse indices from chunks and optional tree nodes."""
        all_texts: list[str] = []
        all_ids: list[str] = []
        all_metadata: list[dict] = []

        # Add document chunks
        for chunk in chunks:
            all_texts.append(chunk.text)
            all_ids.append(chunk.chunk_id)
            all_metadata.append(
                {
                    "type": "chunk",
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.metadata.source,
                    "section": chunk.metadata.section_header,
                    "text": chunk.text,
                }
            )

        # Add tree nodes (all levels for collapsed-tree retrieval)
        if tree:
            for node in tree.get_all_nodes_flat():
                if node.level > 0:  # Skip leaves (already added as chunks)
                    all_texts.append(node.text)
                    all_ids.append(node.node_id)
                    all_metadata.append(
                        {
                            "type": "tree_node",
                            "node_id": node.node_id,
                            "level": node.level,
                            "text": node.text,
                        }
                    )

        if not all_texts:
            logger.warning("No texts to index")
            return

        # Build dense FAISS index
        logger.info(f"Embedding {len(all_texts)} texts...")
        embeddings = self._embed(all_texts)
        self.embedding_dim = embeddings.shape[1]

        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings)

        self.id_to_metadata = {i: meta for i, meta in enumerate(all_metadata)}

        # Build sparse BM25 index
        tokenized = [text.lower().split() for text in all_texts]
        self.bm25 = BM25Okapi(tokenized)
        self.bm25_corpus = all_texts
        self.bm25_ids = all_ids

        logger.info(
            f"Built indices: FAISS={self.faiss_index.ntotal} vectors, "
            f"BM25={len(self.bm25_corpus)} documents"
        )

    def search_dense(
        self,
        query: str,
        top_k: int = settings.faiss_top_k,
        query_embedding: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Dense vector search via FAISS."""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []

        if query_embedding is None:
            query_embedding = self._embed([query])
        elif query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.id_to_metadata.get(int(idx), {})
            results.append(
                {
                    "score": float(score),
                    "rank": len(results) + 1,
                    **meta,
                }
            )
        return results

    def search_sparse(self, query: str, top_k: int = settings.bm25_top_k) -> list[dict]:
        """Sparse BM25 search."""
        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self.id_to_metadata.get(int(idx), {})
            results.append(
                {
                    "score": float(scores[idx]),
                    "rank": len(results) + 1,
                    "bm25_id": self.bm25_ids[idx] if idx < len(self.bm25_ids) else None,
                    **meta,
                }
            )
        return results

    def search_hybrid_embedding(
        self,
        query_embedding: np.ndarray,
        hyde_embedding: Optional[np.ndarray],
        top_k: int = settings.faiss_top_k,
    ) -> list[dict]:
        """Search with both original and HyDE embeddings, merging results."""
        results_original = self.search_dense("", top_k=top_k, query_embedding=query_embedding)

        if hyde_embedding is not None:
            results_hyde = self.search_dense("", top_k=top_k, query_embedding=hyde_embedding)
            # Merge by taking the best score per document
            seen = {}
            for r in results_original:
                key = r.get("chunk_id") or r.get("node_id", "")
                seen[key] = r

            for r in results_hyde:
                key = r.get("chunk_id") or r.get("node_id", "")
                if key in seen:
                    seen[key]["score"] = max(seen[key]["score"], r["score"])
                else:
                    seen[key] = r

            merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
            return merged[:top_k]

        return results_original

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string and return the vector."""
        return self._embed([query])[0]

    def save(self, path: Optional[Path] = None) -> None:
        """Persist both indices to disk."""
        path = path or settings.faiss_index_path
        path.mkdir(parents=True, exist_ok=True)

        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(path / "index.faiss"))

        with open(path / "metadata.json", "w") as f:
            # Convert int keys to string for JSON
            serializable = {str(k): v for k, v in self.id_to_metadata.items()}
            json.dump(serializable, f)

        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "corpus": self.bm25_corpus,
                    "ids": self.bm25_ids,
                },
                f,
            )

        logger.info(f"Saved indices to {path}")

    def load(self, path: Optional[Path] = None) -> None:
        """Load indices from disk."""
        path = path or settings.faiss_index_path

        faiss_path = path / "index.faiss"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))

        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                raw = json.load(f)
                self.id_to_metadata = {int(k): v for k, v in raw.items()}

        bm25_path = path / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.bm25_corpus = data["corpus"]
                self.bm25_ids = data["ids"]

        total = self.faiss_index.ntotal if self.faiss_index else 0
        logger.info(f"Loaded indices from {path}: {total} vectors")

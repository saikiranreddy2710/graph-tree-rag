"""
HyDE — Hypothetical Document Embeddings for zero-shot dense retrieval.
Generates a hypothetical ideal answer document and uses its embedding
to expand the retrieval surface beyond the original query.
Based on: arXiv:2212.10496 (HyDE)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import litellm
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings

logger = logging.getLogger(__name__)

HYDE_PROMPT = (
    Path(__file__).parent.parent / "config" / "prompts" / "hyde_generation.txt"
).read_text()


@dataclass
class HyDEResult:
    hypothetical_document: str
    query_embedding: np.ndarray
    hyde_embedding: np.ndarray
    combined_embedding: np.ndarray  # Weighted average


class HyDEAugmenter:
    """Generates hypothetical document embeddings to improve retrieval recall."""

    def __init__(
        self,
        model: str = settings.fast_model,
        hyde_weight: float = 0.6,
    ):
        self.model = model
        self.hyde_weight = hyde_weight  # Weight for HyDE embedding vs query embedding
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(settings.embedding_model)
        return self._embedder

    def _embed(self, text: str) -> np.ndarray:
        embedder = self._get_embedder()
        return embedder.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        prompt = HYDE_PROMPT.replace("{query}", query)

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=512,
        )
        return response.choices[0].message.content

    async def augment(self, query: str) -> HyDEResult:
        """Generate HyDE embeddings for a query."""
        # Generate hypothetical document
        hypothetical = await self._generate_hypothetical(query)
        logger.debug(f"HyDE generated: {hypothetical[:100]}...")

        # Embed both query and hypothetical document
        query_embedding = self._embed(query)
        hyde_embedding = self._embed(hypothetical)

        # Create weighted combined embedding
        combined = (1 - self.hyde_weight) * query_embedding + self.hyde_weight * hyde_embedding
        # Normalize the combined embedding
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return HyDEResult(
            hypothetical_document=hypothetical,
            query_embedding=query_embedding,
            hyde_embedding=hyde_embedding,
            combined_embedding=combined,
        )

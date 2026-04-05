"""
TurboQuant — MSE-optimal vector quantization for CortexStore embeddings.

Implements the TurboQuant MSE-optimal algorithm (Algorithm 1) from:
  - TurboQuant: arXiv:2504.19874 (Zandieh et al., ICLR 2026)
  - PolarQuant: arXiv:2502.02617 (Han et al., AISTATS 2026)
  - 0xSero/turboquant: GPU-validated implementation with Lloyd-Max codebooks

Algorithm:
    1. Separate vector norm (stored exactly in float32)
    2. Randomly rotate the unit vector using a Haar-distributed orthogonal matrix
    3. Each coordinate of the rotated unit vector follows a Beta distribution
       (≈ Gaussian N(0, 1/d) for large d)
    4. Quantize each coordinate independently using Lloyd-Max optimal codebook
    5. Reconstruct: look up centroids → inverse rotation → rescale by norm

Design decisions:
    - QJL Stage 2 is deliberately OMITTED. For normalized sentence embeddings
      used in RAG retrieval, QJL residual correction introduces noise that
      degrades Pearson correlation from 0.996 to 0.48 (validated empirically
      by Manoj Krishna Mohan, turboqvec). Stage 2 is designed for KV cache
      attention vectors, not embedding retrieval.
    - Lloyd-Max codebooks are computed at init time using the analytical
      coordinate distribution of uniform points on the unit sphere.
    - The random rotation uses the Haar measure correction to ensure
      uniform distribution over the orthogonal group O(d).

Expected quality (d=384, normalized embeddings):
    - 4-bit: ~0.004 mean cosine error, ~95% Recall@1
    - 3-bit: ~0.011 mean cosine error, ~87% Recall@1
    - 2-bit: ~0.031 mean cosine error, ~73% Recall@1
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Lloyd-Max Optimal Scalar Quantizer ────────────────────────────────


def _std_norm_pdf(x: float) -> float:
    """Standard normal PDF φ(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _std_norm_cdf(x: float) -> float:
    """Standard normal CDF Φ(x) via math.erf (stdlib, no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_lloyd_max_codebook(
    sigma: float,
    n_levels: int,
    n_iterations: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the optimal Lloyd-Max codebook for N(0, σ²).

    The Lloyd-Max algorithm minimizes MSE for scalar quantization of a
    known distribution. For N(0, σ²), the conditional expectation
    E[X | a < X < b] has the closed form:

        σ · [φ(a/σ) − φ(b/σ)] / [Φ(b/σ) − Φ(a/σ)]

    This uses only math.erf (stdlib) — no scipy dependency required.

    Args:
        sigma: Standard deviation of the Gaussian distribution.
        n_levels: Number of quantization levels (2^bits).
        n_iterations: Max iterations for convergence.

    Returns:
        centroids: Sorted array of n_levels centroid values.
        boundaries: Array of n_levels - 1 decision boundaries.
    """
    # Initialize centroids evenly in [-3σ, 3σ]
    centroids = np.linspace(-3.0 * sigma, 3.0 * sigma, n_levels).astype(np.float64)

    for iteration in range(n_iterations):
        # Boundaries are midpoints between consecutive centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Extended boundaries: first bin starts at -∞, last bin ends at +∞
        lo = np.concatenate([[-30.0 * sigma], boundaries])
        hi = np.concatenate([boundaries, [30.0 * sigma]])

        new_centroids = np.zeros(n_levels, dtype=np.float64)
        for k in range(n_levels):
            # P(lo < X < hi) for X ~ N(0, σ²)
            p = _std_norm_cdf(hi[k] / sigma) - _std_norm_cdf(lo[k] / sigma)
            if p > 1e-15:
                # E[X | lo < X < hi] = σ · [φ(lo/σ) − φ(hi/σ)] / [Φ(hi/σ) − Φ(lo/σ)]
                pdf_lo = _std_norm_pdf(lo[k] / sigma)
                pdf_hi = _std_norm_pdf(hi[k] / sigma)
                new_centroids[k] = sigma * (pdf_lo - pdf_hi) / p
            else:
                new_centroids[k] = (lo[k] + hi[k]) / 2.0

        if np.allclose(centroids, new_centroids, atol=1e-12):
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.astype(np.float32), boundaries.astype(np.float32)


# ── TurboQuant Compressor ─────────────────────────────────────────────


class TurboQuantCompressor:
    """
    MSE-optimal vector quantizer based on TurboQuant Algorithm 1.

    Compresses d-dimensional float32 vectors to (norm, indices) pairs:
        - norm: float32 scalar (stored exactly)
        - indices: uint8 array of d codebook indices

    Achieves near-optimal distortion (within 2.7× of information-theoretic
    lower bound) at any bit width, with zero dataset preprocessing.

    Storage per vector at d=384:
        - Original:  1536 bytes (384 × float32)
        - 4-bit TQ:   388 bytes (384 × uint8 + float32 norm) = 3.96× reduction
        - With nibble packing: 196 bytes = 7.84× reduction (not yet implemented)
    """

    def __init__(self, d: int, bits: int = 4, seed: int = 42):
        """
        Initialize TurboQuant compressor.

        Args:
            d: Embedding dimension (e.g. 384 for all-MiniLM-L6-v2).
            bits: Quantization bit width (1-8). Default 4 for best quality/size.
            seed: Random seed for reproducible rotation matrix.
        """
        if bits < 1 or bits > 8:
            raise ValueError(f"Bit width must be 1-8, got {bits}")

        self.d = d
        self.bits = bits
        n_levels = 1 << bits  # 2^bits

        # Coordinate std dev for uniform point on unit sphere S^{d-1}
        # Each coordinate ~ Beta distribution ≈ N(0, 1/d) for large d
        # (Lemma 1, TurboQuant paper)
        sigma = 1.0 / math.sqrt(d)

        # Compute optimal Lloyd-Max codebook for this distribution
        logger.info(
            f"Computing Lloyd-Max codebook: d={d}, bits={bits}, "
            f"levels={n_levels}, σ={sigma:.6f}"
        )
        self.centroids, self.boundaries = compute_lloyd_max_codebook(sigma, n_levels)

        # Random orthogonal rotation matrix (Haar-distributed on O(d))
        # QR of Gaussian matrix + sign correction ensures uniform distribution
        # (Section 2.2, PolarQuant paper; rotation.py in 0xSero/turboquant)
        rng = np.random.RandomState(seed)
        H = rng.randn(d, d).astype(np.float64)
        Q, R = np.linalg.qr(H)
        # Haar measure correction: multiply by sign of diagonal of R
        D = np.diag(np.sign(np.diag(R)))
        self.rotation_matrix = (Q @ D).astype(np.float32)

        logger.info(
            f"TurboQuant initialized: {n_levels} centroids in "
            f"[{self.centroids[0]:.6f}, {self.centroids[-1]:.6f}]"
        )

    def compress(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compress float32 embeddings to (norms, indices).

        Args:
            embeddings: float32 array of shape (N, d) or (d,).

        Returns:
            norms: float32 array of shape (N,) — exact L2 norms.
            indices: uint8 array of shape (N, d) — codebook indices per coordinate.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # 1. Separate norm from direction (Section 3.1, TurboQuant paper)
        norms = np.linalg.norm(embeddings, axis=1).astype(np.float32)
        unit = embeddings / (norms[:, np.newaxis] + 1e-10)

        # 2. Random rotation — decorrelates coordinates, induces Beta distribution
        rotated = unit @ self.rotation_matrix.T

        # 3. Quantize each coordinate using Lloyd-Max codebook
        # searchsorted finds the bin index for each value via the boundaries
        indices = np.searchsorted(self.boundaries, rotated).astype(np.uint8)

        return norms, indices

    def decompress(self, norms: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Reconstruct float32 embeddings from compressed representation.

        Args:
            norms: float32 array of shape (N,) or scalar.
            indices: uint8 array of shape (N, d) or (d,).

        Returns:
            Reconstructed float32 embeddings of shape (N, d).
        """
        single = False
        if norms.ndim == 0:
            norms = norms.reshape(1)
            indices = indices.reshape(1, -1)
            single = True
        elif indices.ndim == 1:
            norms = np.atleast_1d(norms)
            indices = indices.reshape(1, -1)
            single = True

        # 1. Look up centroid values from codebook
        reconstructed = self.centroids[indices]

        # 2. Inverse rotation (Π^T = Π^{-1} for orthogonal Π)
        rotated_back = reconstructed @ self.rotation_matrix

        # 3. Rescale by stored norms
        result = rotated_back * norms[:, np.newaxis]

        if single:
            return result[0].astype(np.float32)
        return result.astype(np.float32)

    # ── Diagnostics ───────────────────────────────────────────────

    def measure_quality(self, embeddings: np.ndarray) -> dict:
        """
        Measure compression quality on a set of embeddings.

        Returns dict with MSE, mean cosine error, and theoretical bounds.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        norms, indices = self.compress(embeddings)
        reconstructed = self.decompress(norms, indices)

        # MSE
        mse = float(np.mean(np.sum((embeddings - reconstructed) ** 2, axis=-1)))

        # Cosine similarity
        dot = np.sum(embeddings * reconstructed, axis=-1)
        norm_o = np.linalg.norm(embeddings, axis=-1)
        norm_r = np.linalg.norm(reconstructed, axis=-1)
        cos_sim = dot / (norm_o * norm_r + 1e-10)
        mean_cos_error = float(1.0 - np.mean(cos_sim))

        # Theoretical MSE bound from paper (Theorem 1)
        theoretical_mse = 3.0 * math.pi / 2.0 * (1.0 / (4 ** self.bits))

        return {
            "mse": mse,
            "mean_cosine_error": mean_cos_error,
            "mean_cosine_similarity": float(np.mean(cos_sim)),
            "theoretical_mse_bound": theoretical_mse,
            "compression_ratio": (self.d * 4) / (self.d + 4),  # float32 vs uint8+norm
            "bits": self.bits,
            "dimension": self.d,
        }

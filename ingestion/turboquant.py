"""
TurboQuant - Advanced Vector Compression module for CortexStore.

Based on extreme compression techniques mimicking PolarQuant and QJL:
- PolarQuant: Converts Cartesian vectors to Polar shorthand pairs to drop overhead.
- QJL: Quantized Johnson-Lindenstrauss Transform for 1-bit residual compression.

This allows CortexStore to compress 32-bit floats down to compact 8-bit integers,
enabling scaling to 1M+ nodes without overwhelming memory.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PolarQuantTransformer:
    """
    Transforms standard Cartesian vectors into Polar shorthand.
    Pairs dimensions (x, y) into (r, theta) thereby exploiting geometric
    properties to avoid standard data normalization overhead.
    """

    def __init__(self, d: int, seed: int = 42):
        self.d = d
        np.random.seed(seed)
        # Generate a random orthogonal matrix for the initial rotation
        # Simplifies data geometry before polar conversion as per the paper
        H = np.random.randn(d, d)
        Q, R = np.linalg.qr(H)
        self.rotation_matrix = Q.astype(np.float32)

    def group_to_polar(self, x: np.ndarray) -> np.ndarray:
        """
        Convert pairs of (x, y) coordinates to (radius, angle).
        Input shape: (N, d). Output shape: (N, d).
        """
        # Ensure even dimension padding if needed, though usually embeddings are 384/768
        assert self.d % 2 == 0, "Dimension must be even for PolarQuant"

        N = x.shape[0]
        # 1. Randomly rotate the data to simplify geometry
        x_rot = x @ self.rotation_matrix.T

        # 2. Reshape to pairs: (N, d/2, 2)
        pairs = x_rot.reshape(N, self.d // 2, 2)

        x_coords = pairs[:, :, 0]
        y_coords = pairs[:, :, 1]

        # 3. Cartesian to Polar
        radii = np.sqrt(x_coords**2 + y_coords**2)
        angles = np.arctan2(y_coords, x_coords)

        # 4. Pack back into a flat array: [r1, a1, r2, a2...]
        polar = np.empty_like(x_rot)
        polar[:, 0::2] = radii
        polar[:, 1::2] = angles

        return polar

    def polar_to_group(self, polar: np.ndarray) -> np.ndarray:
        """
        Reconstruct Cartesian vectors from Polar shorthand.
        """
        N = polar.shape[0]
        radii = polar[:, 0::2]
        angles = polar[:, 1::2]

        x_coords = radii * np.cos(angles)
        y_coords = radii * np.sin(angles)

        pairs = np.empty((N, self.d // 2, 2), dtype=np.float32)
        pairs[:, :, 0] = x_coords
        pairs[:, :, 1] = y_coords

        x_rot = pairs.reshape(N, self.d)
        # Inverse rotation (Q constraint means Q.T is Q^-1)
        x = x_rot @ self.rotation_matrix
        return x


class QJLTransformer:
    """
    Quantized Johnson-Lindenstrauss (QJL) Transform.
    Applies the JL transform to the residual error and takes a 1-bit sign.
    """

    def __init__(self, d: int, proj_dim: Optional[int] = None, seed: int = 42):
        self.d = d
        # Project into same dims if None, or a target dimensionality
        self.proj_dim = proj_dim or d
        # FAISS IndexBinaryFlat requires dimensions to be a multiple of 8
        if self.proj_dim % 8 != 0:
            self.proj_dim = ((self.proj_dim // 8) + 1) * 8
            
        np.random.seed(seed + 1)
        # Gaussian random projection matrix
        self.jl_matrix = np.random.randn(self.proj_dim, self.d).astype(np.float32) / np.sqrt(self.proj_dim)

    def compress_residual(self, residual: np.ndarray) -> np.ndarray:
        """
        Project the residual and quantize to 1-bit (+1 or -1).
        Returned as packed bits (uint8) for advanced FAISS Binary integration.
        Input shape: (N, d), Output shape: (N, proj_dim // 8) uint8
        """
        projected = residual @ self.jl_matrix.T
        # 1-bit sign: 1 if >= 0 else 0
        signs = np.where(projected >= 0, 1, 0).astype(np.uint8)
        packed = np.packbits(signs, axis=-1)
        return packed

    def reconstruct_residual(self, packed_signs: np.ndarray) -> np.ndarray:
        """
        Estimate the residual from the packed 1-bit signs.
        """
        signs = np.unpackbits(packed_signs, axis=-1)
        # Restore to +1 / -1
        signs_float = np.where(signs > 0, 1, -1).astype(np.float32)
        return signs_float @ self.jl_matrix


class TurboQuantCompressor:
    """
    Unified high-level interface combining PolarQuant and QJL.
    Compresses 32-bit floats into heavily quantized Int8 byte structures.
    """

    def __init__(self, d: int, bits: int = 3, seed: int = 42):
        self.d = d
        self.bits = bits
        self.polar_quant = PolarQuantTransformer(d=d, seed=seed)
        self.qjl = QJLTransformer(d=d, seed=seed)

        # Calculate dynamic range for linear quantization of polar values (simplified)
        self.quant_max = (1 << (self.bits - 1)) - 1  # e.g. 3 bits -> max 3
        self.quant_min = -(1 << (self.bits - 1))     # e.g. 3 bits -> min -4

    def compress(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compress float32 embeddings to (polar_q, qjl_signs).
        Returns:
            polar_q: int8 array of quantized polar coordinates
            qjl_signs: int8 array of residual signs
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # 1. Transform to Polar
        polar = self.polar_quant.group_to_polar(embeddings)

        # 2. Quantize the polar representation (lossy stage)
        # Scale range roughly from -PI to PI
        scale_factor = np.pi / self.quant_max
        polar_q = np.clip(np.round(polar / scale_factor), self.quant_min, self.quant_max).astype(np.int8)

        # 3. Calculate residual (Reconstruct and compare)
        polar_recon = polar_q.astype(np.float32) * scale_factor
        embeddings_recon = self.polar_quant.polar_to_group(polar_recon)
        residual = embeddings - embeddings_recon

        # 4. QJL 1-bit compression on residual
        qjl_signs = self.qjl.compress_residual(residual)

        return polar_q, qjl_signs

    def decompress(self, polar_q: np.ndarray, qjl_signs: np.ndarray) -> np.ndarray:
        """
        Reconstruct the float32 embeddings for exact attention scoring or FAISS.
        """
        if polar_q.ndim == 1:
            polar_q = polar_q.reshape(1, -1)
            qjl_signs = qjl_signs.reshape(1, -1)

        scale_factor = np.pi / self.quant_max
        polar_recon = polar_q.astype(np.float32) * scale_factor
        base_recon = self.polar_quant.polar_to_group(polar_recon)

        residual_recon = self.qjl.reconstruct_residual(qjl_signs)

        # Final reconstruction = PolarQuant base + QJL error correction
        final_recon = base_recon + residual_recon
        return final_recon

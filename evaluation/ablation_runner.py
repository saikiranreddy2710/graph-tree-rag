"""
Ablation Runner — Compares vector-only vs graph-only vs tree-only vs hybrid,
and with/without HyDE, CRAG, Speculative RAG to prove hybrid superiority.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    name: str
    vector_enabled: bool = True
    bm25_enabled: bool = True
    graph_enabled: bool = True
    tree_enabled: bool = True
    hyde_enabled: bool = True
    crag_enabled: bool = True
    speculative_enabled: bool = True
    self_rag_enabled: bool = True


# Pre-defined ablation configurations
ABLATION_CONFIGS = [
    AblationConfig(
        name="vector_only",
        bm25_enabled=False,
        graph_enabled=False,
        tree_enabled=False,
        hyde_enabled=False,
        crag_enabled=False,
        speculative_enabled=False,
        self_rag_enabled=False,
    ),
    AblationConfig(
        name="vector_bm25",
        graph_enabled=False,
        tree_enabled=False,
        hyde_enabled=False,
        crag_enabled=False,
        speculative_enabled=False,
        self_rag_enabled=False,
    ),
    AblationConfig(
        name="graph_only",
        vector_enabled=False,
        bm25_enabled=False,
        tree_enabled=False,
        hyde_enabled=False,
        crag_enabled=False,
        speculative_enabled=False,
        self_rag_enabled=False,
    ),
    AblationConfig(
        name="tree_only",
        vector_enabled=False,
        bm25_enabled=False,
        graph_enabled=False,
        hyde_enabled=False,
        crag_enabled=False,
        speculative_enabled=False,
        self_rag_enabled=False,
    ),
    AblationConfig(
        name="hybrid_no_hyde",
        hyde_enabled=False,
        crag_enabled=False,
        speculative_enabled=False,
        self_rag_enabled=False,
    ),
    AblationConfig(
        name="hybrid_no_crag", crag_enabled=False, speculative_enabled=False, self_rag_enabled=False
    ),
    AblationConfig(name="hybrid_no_speculative", speculative_enabled=False, self_rag_enabled=False),
    AblationConfig(name="hybrid_no_self_rag", self_rag_enabled=False),
    AblationConfig(name="full_hybrid"),
]


@dataclass
class AblationResult:
    config_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    num_queries: int = 0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "config": self.config_name,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "num_queries": self.num_queries,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


class AblationRunner:
    """Runs ablation experiments across different configurations."""

    def __init__(self, configs: list[AblationConfig] = None):
        self.configs = configs or ABLATION_CONFIGS

    def format_comparison_table(self, results: list[AblationResult]) -> str:
        """Format results as a comparison table."""
        if not results:
            return "No results"

        # Collect all metric keys
        all_keys = set()
        for r in results:
            all_keys.update(r.metrics.keys())
        sorted_keys = sorted(all_keys)

        # Build table
        header = (
            f"{'Config':<25} " + " ".join(f"{k:>12}" for k in sorted_keys) + f" {'Latency':>10}"
        )
        separator = "-" * len(header)

        rows = [header, separator]
        for r in results:
            values = " ".join(f"{r.metrics.get(k, 0.0):>12.4f}" for k in sorted_keys)
            rows.append(f"{r.config_name:<25} {values} {r.avg_latency_ms:>9.0f}ms")

        return "\n".join(rows)

    def save_results(self, results: list[AblationResult], path: str) -> None:
        """Save ablation results to JSON."""
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info(f"Saved ablation results to {path}")

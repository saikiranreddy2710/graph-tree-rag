"""
Central configuration for Graph-Tree Hybrid RAG.
All settings are overridable via environment variables or .env file.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalStrategy(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="GTR_",
        case_sensitive=False,
    )

    # ── LLM Configuration ──────────────────────────────────────────────
    fast_model: str = Field(
        default="gpt-4o-mini",
        description="Fast model for extraction, routing, drafting",
    )
    strong_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Strong model for verification and final generation",
    )
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, ge=256)
    llm_request_timeout: int = Field(default=60, description="Seconds")

    # ── Embedding Configuration ────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)

    # ── Chunking ───────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, description="Tokens per chunk")
    chunk_overlap: int = Field(default=64, description="Overlap tokens")

    # ── FAISS Vector Store ─────────────────────────────────────────────
    faiss_index_path: Path = Field(default=Path("data/faiss_index"))
    faiss_top_k: int = Field(default=20)

    # ── BM25 Sparse Retrieval ──────────────────────────────────────────
    bm25_top_k: int = Field(default=15)

    # ── Graph Store ────────────────────────────────────────────────────
    graph_persist_path: Path = Field(default=Path("data/knowledge_graph.json"))
    graph_max_hops: int = Field(default=2)
    graph_top_k: int = Field(default=15)
    graph_confidence_threshold: float = Field(
        default=0.5, description="Min confidence for edge traversal"
    )
    leiden_resolution: float = Field(default=1.0)
    community_max_level: int = Field(default=3)

    # ── RAPTOR Tree ────────────────────────────────────────────────────
    tree_persist_path: Path = Field(default=Path("data/raptor_tree.json"))
    tree_max_levels: int = Field(default=4)
    tree_cluster_dim: int = Field(default=10, description="UMAP target dims")
    tree_min_cluster_size: int = Field(default=3)
    tree_top_k: int = Field(default=10)

    # ── Hybrid Fusion ──────────────────────────────────────────────────
    fusion_weights_vector: float = Field(default=0.30)
    fusion_weights_bm25: float = Field(default=0.15)
    fusion_weights_graph: float = Field(default=0.30)
    fusion_weights_tree: float = Field(default=0.25)
    fusion_rrf_k: int = Field(default=60, description="RRF constant k")
    final_top_k: int = Field(default=10)

    # ── Re-ranking ─────────────────────────────────────────────────────
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L6-v2")
    reranker_top_k: int = Field(default=8)
    mmr_lambda: float = Field(default=0.7, description="MMR diversity param")
    graph_coherence_bonus: float = Field(default=0.15)
    temporal_recency_bonus: float = Field(default=0.05)

    # ── HyDE ───────────────────────────────────────────────────────────
    hyde_enabled: bool = Field(default=True)

    # ── CRAG (Corrective RAG) ──────────────────────────────────────────
    crag_enabled: bool = Field(default=True)
    crag_confidence_threshold: float = Field(default=0.6)
    web_search_enabled: bool = Field(default=True)
    web_search_api_key: Optional[str] = Field(default=None, description="Tavily API key")
    web_search_max_results: int = Field(default=5)

    # ── Speculative RAG (Mixture-of-Thought) ───────────────────────────
    speculative_enabled: bool = Field(default=True)
    speculative_num_drafts: int = Field(default=3)
    speculative_draft_model: Optional[str] = Field(
        default=None, description="Override: defaults to fast_model"
    )
    speculative_verifier_model: Optional[str] = Field(
        default=None, description="Override: defaults to strong_model"
    )

    # ── Self-RAG ───────────────────────────────────────────────────────
    self_rag_enabled: bool = Field(default=True)
    self_rag_max_loops: int = Field(default=2)
    self_rag_support_threshold: float = Field(default=0.6)

    # ── Adaptive Router ────────────────────────────────────────────────
    router_default_strategy: RetrievalStrategy = Field(default=RetrievalStrategy.MODERATE)

    # ── Security ───────────────────────────────────────────────────────
    security_prompt_guard_enabled: bool = Field(default=True)
    security_pii_filter_enabled: bool = Field(default=True)
    security_rbac_enabled: bool = Field(default=True)
    security_max_query_length: int = Field(default=2000)
    security_rate_limit_rpm: int = Field(default=60)
    audit_log_path: Path = Field(default=Path("data/audit.jsonl"))

    # ── Observability ──────────────────────────────────────────────────
    tracing_enabled: bool = Field(default=True)
    tracing_service_name: str = Field(default="graph-tree-rag")
    log_level: str = Field(default="INFO")

    # ── Evaluation ─────────────────────────────────────────────────────
    eval_test_queries_path: Path = Field(default=Path("evaluation/test_queries"))
    eval_baseline_path: Path = Field(default=Path("data/eval_baseline.json"))
    eval_regression_threshold: float = Field(default=0.05, description="Max allowed metric drop")

    @property
    def draft_model(self) -> str:
        return self.speculative_draft_model or self.fast_model

    @property
    def verifier_model(self) -> str:
        return self.speculative_verifier_model or self.strong_model


settings = Settings()

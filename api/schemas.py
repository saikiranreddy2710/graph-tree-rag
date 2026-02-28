"""API request/response schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(default="anonymous")
    user_role: str = Field(default="user")
    force_strategy: Optional[str] = Field(
        default=None, description="Override: simple, moderate, complex"
    )
    hyde_enabled: Optional[bool] = Field(default=None)
    speculative_enabled: Optional[bool] = Field(default=None)


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Path to file or directory")
    glob_pattern: str = Field(default="**/*.*")


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    sources: list[dict[str, Any]]
    trace: dict[str, Any]
    timestamp: str
    trace_id: str
    warnings: list[str]


class IngestResponse(BaseModel):
    status: str
    stats: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    indices_loaded: bool
    graph_nodes: int
    tree_nodes: int
    vector_count: int

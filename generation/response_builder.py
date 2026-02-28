"""
Response Builder — Assembles the final response with citations,
provenance chain, confidence scores, and retrieval trace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SourceReference:
    doc_id: str
    text_snippet: str
    source: str
    relevance_score: float
    channel: str  # Which retrieval channel found this
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    router_strategy: str
    router_query_type: str
    sub_questions: list[str] = field(default_factory=list)
    hyde_used: bool = False
    hyde_document: str = ""
    channels_used: list[str] = field(default_factory=list)
    channel_result_counts: dict[str, int] = field(default_factory=dict)
    fusion_method: str = "RRF"
    crag_quality: str = ""
    crag_action: str = ""
    web_search_used: bool = False
    speculative_strategies: list[str] = field(default_factory=list)
    speculative_selected: str = ""
    self_rag_iterations: int = 0
    self_rag_scores: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0


@dataclass
class PipelineResponse:
    """The complete response object returned to the user."""

    query: str
    answer: str
    confidence: float
    sources: list[SourceReference] = field(default_factory=list)
    trace: Optional[RetrievalTrace] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    trace_id: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "sources": [
                {
                    "doc_id": s.doc_id,
                    "text_snippet": s.text_snippet[:200],
                    "source": s.source,
                    "relevance_score": round(s.relevance_score, 3),
                    "channel": s.channel,
                }
                for s in self.sources
            ],
            "trace": {
                "strategy": self.trace.router_strategy if self.trace else "",
                "query_type": self.trace.router_query_type if self.trace else "",
                "channels_used": self.trace.channels_used if self.trace else [],
                "crag_quality": self.trace.crag_quality if self.trace else "",
                "speculative_strategies": (self.trace.speculative_strategies if self.trace else []),
                "self_rag_iterations": (self.trace.self_rag_iterations if self.trace else 0),
                "self_rag_scores": self.trace.self_rag_scores if self.trace else {},
                "latency_ms": self.trace.total_latency_ms if self.trace else 0,
            },
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "warnings": self.warnings,
        }

    def to_display_string(self) -> str:
        """Format for CLI/terminal display."""
        lines = [
            f"Answer: {self.answer}",
            f"\nConfidence: {self.confidence:.1%}",
        ]

        if self.sources:
            lines.append(f"\nSources ({len(self.sources)}):")
            for i, src in enumerate(self.sources, 1):
                lines.append(
                    f"  [{i}] {src.source} (score: {src.relevance_score:.3f}, via: {src.channel})"
                )
                lines.append(f"      {src.text_snippet[:150]}...")

        if self.trace:
            lines.append(f"\nRetrieval Trace:")
            lines.append(f"  Strategy: {self.trace.router_strategy}")
            lines.append(f"  Query Type: {self.trace.router_query_type}")
            lines.append(f"  Channels: {', '.join(self.trace.channels_used)}")
            if self.trace.crag_quality:
                lines.append(f"  CRAG Quality: {self.trace.crag_quality}")
            if self.trace.speculative_strategies:
                lines.append(f"  MoT Strategies: {', '.join(self.trace.speculative_strategies)}")
            lines.append(f"  Self-RAG Iterations: {self.trace.self_rag_iterations}")
            if self.trace.total_latency_ms:
                lines.append(f"  Latency: {self.trace.total_latency_ms:.0f}ms")

        if self.warnings:
            lines.append(f"\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)


class ResponseBuilder:
    """Assembles the final pipeline response with provenance."""

    def build(
        self,
        query: str,
        answer: str,
        context_results: list,
        trace: RetrievalTrace,
        reflection_scores: Optional[dict] = None,
        trace_id: str = "",
    ) -> PipelineResponse:
        """Build the complete response object."""
        # Build source references
        sources = []
        for r in context_results:
            doc_id = getattr(r, "doc_id", "") or ""
            text = getattr(r, "text", "") or ""
            score = getattr(r, "fused_score", 0.0) or 0.0

            # Determine primary channel
            channel_scores = getattr(r, "channel_scores", {})
            primary_channel = (
                max(channel_scores, key=channel_scores.get) if channel_scores else "unknown"
            )

            metadata = getattr(r, "metadata", {})
            source_name = metadata.get("source", doc_id)

            sources.append(
                SourceReference(
                    doc_id=doc_id,
                    text_snippet=text[:300],
                    source=source_name,
                    relevance_score=score,
                    channel=primary_channel,
                    metadata=metadata,
                )
            )

        # Calculate confidence from multiple signals
        confidence = self._calculate_confidence(trace, reflection_scores, len(sources))

        # Build warnings
        warnings = []
        if trace.crag_quality == "INCORRECT":
            warnings.append("Retrieval quality was low; web search fallback was used")
        if trace.crag_quality == "AMBIGUOUS":
            warnings.append("Retrieved context may not fully cover the query")
        if reflection_scores and reflection_scores.get("support", 1.0) < 0.5:
            warnings.append("Some claims may not be fully supported by sources")
        if not sources:
            warnings.append("No sources were found for this query")

        return PipelineResponse(
            query=query,
            answer=answer,
            confidence=confidence,
            sources=sources,
            trace=trace,
            trace_id=trace_id,
            warnings=warnings,
        )

    def _calculate_confidence(
        self,
        trace: RetrievalTrace,
        reflection_scores: Optional[dict],
        num_sources: int,
    ) -> float:
        """Calculate overall confidence from pipeline signals."""
        signals = []

        # CRAG confidence
        if trace.crag_quality == "CORRECT":
            signals.append(0.9)
        elif trace.crag_quality == "AMBIGUOUS":
            signals.append(0.5)
        else:
            signals.append(0.2)

        # Self-RAG reflection scores
        if reflection_scores:
            avg = sum(reflection_scores.values()) / max(len(reflection_scores), 1)
            signals.append(avg)

        # Source availability
        if num_sources >= 5:
            signals.append(0.9)
        elif num_sources >= 2:
            signals.append(0.7)
        elif num_sources >= 1:
            signals.append(0.5)
        else:
            signals.append(0.1)

        return sum(signals) / max(len(signals), 1)

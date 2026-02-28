"""
Audit Logger — Structured audit logging for all pipeline operations.
Logs queries, retrievals, generations, and security events as JSON lines.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class AuditLogger:
    """Structured audit logging to JSONL file."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or settings.audit_log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_entry(self, entry: dict) -> None:
        """Append a JSON entry to the audit log."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def generate_trace_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def log_query(
        self,
        trace_id: str,
        query: str,
        user_id: str = "anonymous",
        user_role: str = "user",
        metadata: Optional[dict] = None,
    ) -> None:
        self._write_entry(
            {
                "event": "query",
                "trace_id": trace_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "user_role": user_role,
                "query": query[:500],
                "metadata": metadata or {},
            }
        )

    def log_security_event(
        self,
        trace_id: str,
        event_type: str,
        details: dict,
        user_id: str = "anonymous",
    ) -> None:
        self._write_entry(
            {
                "event": "security",
                "trace_id": trace_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "event_type": event_type,
                "details": details,
            }
        )

    def log_retrieval(
        self,
        trace_id: str,
        strategy: str,
        num_results: int,
        channels: dict[str, int],
        crag_quality: str = "",
        latency_ms: float = 0.0,
    ) -> None:
        self._write_entry(
            {
                "event": "retrieval",
                "trace_id": trace_id,
                "timestamp": datetime.utcnow().isoformat(),
                "strategy": strategy,
                "num_results": num_results,
                "channels": channels,
                "crag_quality": crag_quality,
                "latency_ms": latency_ms,
            }
        )

    def log_generation(
        self,
        trace_id: str,
        model: str,
        strategy: str,
        num_drafts: int = 0,
        self_rag_iterations: int = 0,
        confidence: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        self._write_entry(
            {
                "event": "generation",
                "trace_id": trace_id,
                "timestamp": datetime.utcnow().isoformat(),
                "model": model,
                "strategy": strategy,
                "num_drafts": num_drafts,
                "self_rag_iterations": self_rag_iterations,
                "confidence": confidence,
                "latency_ms": latency_ms,
            }
        )

    def log_response(
        self,
        trace_id: str,
        query: str,
        answer: str,
        confidence: float,
        total_latency_ms: float,
        num_sources: int,
    ) -> None:
        self._write_entry(
            {
                "event": "response",
                "trace_id": trace_id,
                "timestamp": datetime.utcnow().isoformat(),
                "query": query[:500],
                "answer_length": len(answer),
                "confidence": confidence,
                "total_latency_ms": total_latency_ms,
                "num_sources": num_sources,
            }
        )

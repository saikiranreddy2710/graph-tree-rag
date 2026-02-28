"""
OpenTelemetry Tracer — Distributed tracing for the full pipeline.
Tracks latency, spans, and attributes for each pipeline stage.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SpanRecord:
    name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    children: list[SpanRecord] = field(default_factory=list)


class PipelineTracer:
    """Traces pipeline execution for observability."""

    def __init__(self, service_name: str = settings.tracing_service_name):
        self.service_name = service_name
        self.enabled = settings.tracing_enabled
        self._otel_tracer = None
        self._spans: list[SpanRecord] = []

        if self.enabled:
            self._init_otel()

    def _init_otel(self):
        """Initialize OpenTelemetry tracer."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )

            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            trace.set_tracer_provider(provider)
            self._otel_tracer = trace.get_tracer(self.service_name)
            logger.info("OpenTelemetry tracer initialized")
        except ImportError:
            logger.warning("OpenTelemetry not available, using basic tracing")

    @contextmanager
    def span(
        self, name: str, attributes: Optional[dict] = None
    ) -> Generator[SpanRecord, None, None]:
        """Context manager for tracing a pipeline stage."""
        record = SpanRecord(
            name=name,
            start_time=time.time(),
            attributes=attributes or {},
        )

        if self._otel_tracer:
            with self._otel_tracer.start_as_current_span(name, attributes=attributes or {}):
                try:
                    yield record
                except Exception as e:
                    record.status = f"error: {e}"
                    raise
                finally:
                    record.end_time = time.time()
                    record.duration_ms = (record.end_time - record.start_time) * 1000
                    self._spans.append(record)
        else:
            try:
                yield record
            except Exception as e:
                record.status = f"error: {e}"
                raise
            finally:
                record.end_time = time.time()
                record.duration_ms = (record.end_time - record.start_time) * 1000
                self._spans.append(record)

    def get_trace_summary(self) -> list[dict]:
        """Return a summary of all recorded spans."""
        return [
            {
                "name": s.name,
                "duration_ms": round(s.duration_ms, 2),
                "status": s.status,
                "attributes": s.attributes,
            }
            for s in self._spans
        ]

    def reset(self):
        """Clear recorded spans."""
        self._spans.clear()

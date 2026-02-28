"""
PII Filter — Detects and redacts personally identifiable information
from both input queries and output responses.
Uses regex-based detection (presidio-free fallback available).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PIIDetection:
    entity_type: str
    start: int
    end: int
    text: str
    confidence: float = 0.9


@dataclass
class PIIFilterResult:
    original_text: str
    filtered_text: str
    detections: list[PIIDetection] = field(default_factory=list)
    pii_found: bool = False

    @property
    def num_redactions(self) -> int:
        return len(self.detections)


# Regex patterns for common PII types
PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "PHONE_US": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "DATE_OF_BIRTH": re.compile(
        r"\b(?:DOB|date of birth|born on)[:\s]*\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
        re.IGNORECASE,
    ),
    "AWS_KEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "API_KEY": re.compile(
        r"\b(?:sk-|pk_|rk_|api[_-]?key[=:\s]+)[A-Za-z0-9_-]{20,}\b",
        re.IGNORECASE,
    ),
}

# Redaction placeholders
REDACTION_MAP = {
    "EMAIL": "[EMAIL_REDACTED]",
    "PHONE_US": "[PHONE_REDACTED]",
    "SSN": "[SSN_REDACTED]",
    "CREDIT_CARD": "[CARD_REDACTED]",
    "IP_ADDRESS": "[IP_REDACTED]",
    "DATE_OF_BIRTH": "[DOB_REDACTED]",
    "AWS_KEY": "[AWS_KEY_REDACTED]",
    "API_KEY": "[API_KEY_REDACTED]",
}


class PIIFilter:
    """Detects and redacts PII from text using regex patterns."""

    def __init__(
        self,
        enabled: bool = settings.security_pii_filter_enabled,
        use_presidio: bool = False,
    ):
        self.enabled = enabled
        self.use_presidio = use_presidio
        self._presidio_analyzer = None
        self._presidio_anonymizer = None

        if use_presidio:
            self._init_presidio()

    def _init_presidio(self):
        """Initialize Presidio analyzer and anonymizer."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._presidio_analyzer = AnalyzerEngine()
            self._presidio_anonymizer = AnonymizerEngine()
            logger.info("Presidio PII detection initialized")
        except ImportError:
            logger.warning("Presidio not available, using regex-only PII detection")
            self.use_presidio = False

    def _detect_regex(self, text: str) -> list[PIIDetection]:
        """Detect PII using regex patterns."""
        detections = []
        for pii_type, pattern in PII_PATTERNS.items():
            for match in pattern.finditer(text):
                detections.append(
                    PIIDetection(
                        entity_type=pii_type,
                        start=match.start(),
                        end=match.end(),
                        text=match.group(),
                        confidence=0.85,
                    )
                )
        return detections

    def _detect_presidio(self, text: str) -> list[PIIDetection]:
        """Detect PII using Presidio NER."""
        if not self._presidio_analyzer:
            return []

        results = self._presidio_analyzer.analyze(
            text=text,
            language="en",
            entities=[
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "US_SSN",
                "IP_ADDRESS",
                "PERSON",
                "LOCATION",
            ],
        )

        return [
            PIIDetection(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                text=text[r.start : r.end],
                confidence=r.score,
            )
            for r in results
        ]

    def _redact(self, text: str, detections: list[PIIDetection]) -> str:
        """Replace detected PII with redaction placeholders."""
        if not detections:
            return text

        # Sort by position (reverse) to replace from end to start
        sorted_detections = sorted(detections, key=lambda d: d.start, reverse=True)
        result = text

        for det in sorted_detections:
            placeholder = REDACTION_MAP.get(det.entity_type, f"[{det.entity_type}_REDACTED]")
            result = result[: det.start] + placeholder + result[det.end :]

        return result

    def filter_text(self, text: str) -> PIIFilterResult:
        """Detect and redact PII from text."""
        if not self.enabled:
            return PIIFilterResult(original_text=text, filtered_text=text, pii_found=False)

        # Run detection
        detections = self._detect_regex(text)
        if self.use_presidio:
            detections.extend(self._detect_presidio(text))

        # Deduplicate overlapping detections
        detections = self._deduplicate(detections)

        if detections:
            filtered = self._redact(text, detections)
            logger.info(
                f"PII filter: redacted {len(detections)} items "
                f"({set(d.entity_type for d in detections)})"
            )
            return PIIFilterResult(
                original_text=text,
                filtered_text=filtered,
                detections=detections,
                pii_found=True,
            )

        return PIIFilterResult(original_text=text, filtered_text=text, pii_found=False)

    def _deduplicate(self, detections: list[PIIDetection]) -> list[PIIDetection]:
        """Remove overlapping detections, keeping higher confidence."""
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda d: (d.start, -d.confidence))
        result = [sorted_dets[0]]

        for det in sorted_dets[1:]:
            last = result[-1]
            if det.start >= last.end:
                result.append(det)
            elif det.confidence > last.confidence:
                result[-1] = det

        return result

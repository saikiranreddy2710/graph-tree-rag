"""
Prompt Injection Guard — Detects and blocks prompt injection attacks.
Uses regex patterns + heuristic analysis to identify malicious queries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Known injection patterns
INJECTION_PATTERNS = [
    # Direct instruction override
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
    r"forget\s+(all\s+)?(previous|above|prior)\s+(instructions?|context)",
    # System prompt extraction
    r"(print|show|reveal|display|output|repeat)\s+(your\s+)?(system\s+)?(prompt|instructions?|rules?)",
    r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)",
    # Role play attacks
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"act\s+as\s+(a|an|the)\s+",
    r"pretend\s+(to\s+be|you\s+are)\s+",
    r"switch\s+to\s+.*mode",
    # Jailbreak patterns
    r"DAN\s+mode",
    r"developer\s+mode",
    r"do\s+anything\s+now",
    r"bypass\s+(your\s+)?(safety|filter|restriction|guard)",
    # Encoding attacks
    r"base64\s*:",
    r"\\x[0-9a-fA-F]{2}",
    # Delimiter injection
    r"<\|.*\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
]

# Suspicious token patterns
SUSPICIOUS_TOKENS = [
    "```system",
    "```instruction",
    "<system>",
    "</system>",
    "ADMIN:",
    "ROOT:",
    "OVERRIDE:",
    "SUDO:",
]


@dataclass
class GuardResult:
    is_safe: bool
    risk_score: float  # 0.0 (safe) to 1.0 (dangerous)
    blocked_reason: Optional[str] = None
    sanitized_query: Optional[str] = None
    matched_patterns: list[str] = None

    def __post_init__(self):
        if self.matched_patterns is None:
            self.matched_patterns = []


class PromptGuard:
    """Detects and blocks prompt injection attacks in user queries."""

    def __init__(self, enabled: bool = settings.security_prompt_guard_enabled):
        self.enabled = enabled
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

    def _check_patterns(self, query: str) -> list[str]:
        """Check query against known injection patterns."""
        matched = []
        for pattern in self._compiled_patterns:
            if pattern.search(query):
                matched.append(pattern.pattern[:50])
        return matched

    def _check_suspicious_tokens(self, query: str) -> list[str]:
        """Check for suspicious delimiter/system tokens."""
        found = []
        query_lower = query.lower()
        for token in SUSPICIOUS_TOKENS:
            if token.lower() in query_lower:
                found.append(token)
        return found

    def _check_length(self, query: str) -> bool:
        """Flag excessively long queries (possible injection padding)."""
        return len(query) > settings.security_max_query_length

    def _calculate_risk_score(
        self,
        pattern_matches: list[str],
        token_matches: list[str],
        is_too_long: bool,
    ) -> float:
        """Calculate a composite risk score."""
        score = 0.0
        # Even a single pattern match is highly suspicious (0.5 = block threshold)
        score += min(len(pattern_matches) * 0.5, 0.95)
        score += min(len(token_matches) * 0.2, 0.5)
        if is_too_long:
            score += 0.1
        return min(score, 1.0)

    def _sanitize(self, query: str) -> str:
        """Strip known dangerous patterns from the query."""
        sanitized = query
        # Remove control characters
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", sanitized)
        # Remove delimiter injections
        sanitized = re.sub(r"<\|.*?\|>", "", sanitized)
        sanitized = re.sub(r"\[INST\]|\[/INST\]", "", sanitized)
        sanitized = re.sub(r"<<SYS>>|<</SYS>>", "", sanitized)
        # Truncate if too long
        if len(sanitized) > settings.security_max_query_length:
            sanitized = sanitized[: settings.security_max_query_length]
        return sanitized.strip()

    def check(self, query: str) -> GuardResult:
        """Check a query for prompt injection attacks."""
        if not self.enabled:
            return GuardResult(is_safe=True, risk_score=0.0, sanitized_query=query)

        pattern_matches = self._check_patterns(query)
        token_matches = self._check_suspicious_tokens(query)
        is_too_long = self._check_length(query)
        risk_score = self._calculate_risk_score(pattern_matches, token_matches, is_too_long)

        all_matches = pattern_matches + token_matches

        if risk_score >= 0.5:
            logger.warning(
                f"Prompt injection BLOCKED (risk={risk_score:.2f}): "
                f"patterns={pattern_matches}, tokens={token_matches}"
            )
            return GuardResult(
                is_safe=False,
                risk_score=risk_score,
                blocked_reason=(f"Query blocked: detected {len(all_matches)} injection pattern(s)"),
                matched_patterns=all_matches,
            )

        # Sanitize and allow
        sanitized = self._sanitize(query) if risk_score > 0.1 else query

        if risk_score > 0.1:
            logger.info(f"Prompt guard: low risk ({risk_score:.2f}), sanitized query")

        return GuardResult(
            is_safe=True,
            risk_score=risk_score,
            sanitized_query=sanitized,
            matched_patterns=all_matches,
        )

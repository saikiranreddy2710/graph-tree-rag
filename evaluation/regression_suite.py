"""
Regression Suite — Automated test runner to catch quality regressions
across prompt/retrieval changes. Compares current metrics against saved baselines.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from evaluation.metrics import EvalResult, MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class TestQuery:
    query: str
    gold_answer: str
    gold_source_ids: list[str] = field(default_factory=list)
    gold_entities: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"  # factoid, comparison, multi-hop, etc.
    tags: list[str] = field(default_factory=list)


@dataclass
class RegressionResult:
    passed: bool
    total_queries: int
    passed_queries: int
    failed_queries: int
    metric_diffs: dict[str, float] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total_queries": self.total_queries,
            "passed_queries": self.passed_queries,
            "failed_queries": self.failed_queries,
            "metric_diffs": self.metric_diffs,
            "failures": self.failures,
            "summary": self.summary,
        }


class RegressionSuite:
    """Runs regression tests and compares against baseline metrics."""

    def __init__(
        self,
        test_queries_path: Optional[Path] = None,
        baseline_path: Optional[Path] = None,
        threshold: float = settings.eval_regression_threshold,
    ):
        self.test_queries_path = test_queries_path or settings.eval_test_queries_path
        self.baseline_path = baseline_path or settings.eval_baseline_path
        self.threshold = threshold
        self.calculator = MetricsCalculator()

    def load_test_queries(self) -> list[TestQuery]:
        """Load test queries from JSON files."""
        queries = []
        path = self.test_queries_path

        if path.is_file():
            with open(path) as f:
                data = json.load(f)
                for q in data:
                    queries.append(TestQuery(**q))
        elif path.is_dir():
            for file in sorted(path.glob("*.json")):
                with open(file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for q in data:
                            queries.append(TestQuery(**q))
                    else:
                        queries.append(TestQuery(**data))

        logger.info(f"Loaded {len(queries)} test queries")
        return queries

    def load_baseline(self) -> Optional[dict]:
        """Load baseline metrics from a previous run."""
        if not self.baseline_path.exists():
            return None
        with open(self.baseline_path) as f:
            return json.load(f)

    def save_baseline(self, results: list[EvalResult]) -> None:
        """Save current results as the new baseline."""
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)

        # Average metrics across all queries
        avg_metrics = self._average_metrics(results)
        with open(self.baseline_path, "w") as f:
            json.dump(avg_metrics, f, indent=2)
        logger.info(f"Saved baseline to {self.baseline_path}")

    def _average_metrics(self, results: list[EvalResult]) -> dict:
        """Compute average metrics across all evaluation results."""
        if not results:
            return {}

        retrieval_keys = [
            "precision@10",
            "recall@10",
            "mrr",
            "ndcg@10",
            "hit_rate",
            "entity_coverage",
        ]
        generation_keys = [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "answer_similarity",
        ]

        avg = {}
        for key in retrieval_keys:
            values = [r.retrieval.to_dict().get(key, 0) for r in results]
            avg[f"retrieval_{key}"] = sum(values) / len(values)

        for key in generation_keys:
            values = [r.generation.to_dict().get(key, 0) for r in results]
            avg[f"generation_{key}"] = sum(values) / len(values)

        return avg

    def compare_with_baseline(self, results: list[EvalResult]) -> RegressionResult:
        """Compare current results against baseline, detect regressions."""
        baseline = self.load_baseline()
        current = self._average_metrics(results)

        if baseline is None:
            logger.info("No baseline found, saving current as baseline")
            self.save_baseline(results)
            return RegressionResult(
                passed=True,
                total_queries=len(results),
                passed_queries=len(results),
                failed_queries=0,
                summary="First run — saved as baseline",
            )

        # Compare each metric
        diffs = {}
        failures = []
        for key in baseline:
            old_val = baseline.get(key, 0)
            new_val = current.get(key, 0)
            diff = new_val - old_val
            diffs[key] = round(diff, 4)

            if diff < -self.threshold:
                failures.append(
                    {
                        "metric": key,
                        "baseline": round(old_val, 4),
                        "current": round(new_val, 4),
                        "diff": round(diff, 4),
                        "threshold": self.threshold,
                    }
                )

        passed = len(failures) == 0
        summary_parts = []
        if passed:
            summary_parts.append("All metrics within threshold")
        else:
            summary_parts.append(f"REGRESSION: {len(failures)} metric(s) dropped below threshold")
            for f in failures:
                summary_parts.append(
                    f"  - {f['metric']}: {f['baseline']} -> {f['current']} "
                    f"(delta={f['diff']}, threshold=-{self.threshold})"
                )

        return RegressionResult(
            passed=passed,
            total_queries=len(results),
            passed_queries=len(results) - len(failures),
            failed_queries=len(failures),
            metric_diffs=diffs,
            failures=failures,
            summary="\n".join(summary_parts),
        )

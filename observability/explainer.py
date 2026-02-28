"""
Retrieval Explainer — Generates human-readable explanations of
the retrieval pipeline decisions for any query.
"""

from __future__ import annotations

from generation.response_builder import PipelineResponse, RetrievalTrace


class RetrievalExplainer:
    """Generates human-readable explanations for retrieval decisions."""

    def explain(self, response: PipelineResponse) -> str:
        """Generate a step-by-step explanation of the retrieval pipeline."""
        if not response.trace:
            return "No trace information available."

        trace = response.trace
        steps = []

        # Step 1: Routing
        steps.append(
            f"1. ROUTING: Query classified as [{trace.router_strategy}] "
            f"(type: {trace.router_query_type})"
        )

        if trace.sub_questions:
            steps.append(f"   - Decomposed into {len(trace.sub_questions)} sub-questions:")
            for sq in trace.sub_questions:
                steps.append(f"     * {sq}")

        # Step 2: HyDE
        if trace.hyde_used:
            steps.append(
                f'2. HyDE: Generated hypothetical answer document ("{trace.hyde_document[:80]}...")'
            )
        else:
            steps.append("2. HyDE: Skipped (not enabled for this query type)")

        # Step 3: Retrieval
        steps.append(f"3. RETRIEVAL: Used {len(trace.channels_used)} channels:")
        for ch, count in (trace.channel_result_counts or {}).items():
            steps.append(f"   - {ch}: {count} results")

        # Step 4: CRAG
        if trace.crag_quality:
            steps.append(
                f"4. CRAG EVALUATION: Quality={trace.crag_quality}, Action={trace.crag_action}"
            )
            if trace.web_search_used:
                steps.append("   - Web search fallback was triggered")
        else:
            steps.append("4. CRAG: Skipped")

        # Step 5: Speculative RAG
        if trace.speculative_strategies:
            steps.append(
                f"5. SPECULATIVE RAG (MoT): Generated {len(trace.speculative_strategies)} drafts:"
            )
            for strategy in trace.speculative_strategies:
                marker = " <-- SELECTED" if strategy == trace.speculative_selected else ""
                steps.append(f"   - {strategy}{marker}")
        else:
            steps.append("5. SPECULATIVE RAG: Skipped")

        # Step 6: Self-RAG
        if trace.self_rag_iterations > 0:
            steps.append(f"6. SELF-RAG: {trace.self_rag_iterations} reflection iteration(s)")
            for key, val in (trace.self_rag_scores or {}).items():
                steps.append(f"   - {key}: {val:.2f}")
        else:
            steps.append("6. SELF-RAG: Skipped")

        # Step 7: Final
        steps.append(
            f"7. RESULT: Confidence={response.confidence:.1%}, "
            f"{len(response.sources)} sources, "
            f"latency={trace.total_latency_ms:.0f}ms"
        )

        if response.warnings:
            steps.append("   WARNINGS:")
            for w in response.warnings:
                steps.append(f"   - {w}")

        return "\n".join(steps)

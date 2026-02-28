"""
Speculative RAG with Mixture-of-Thought — Parallel draft generation from
document subsets using diverse reasoning strategies, verified by a strong model.
Based on: Speculative RAG (ICLR 2025, arXiv:2407.08223)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from retrieval.hybrid_fuser import FusedResult

logger = logging.getLogger(__name__)

DRAFT_PROMPT = (
    Path(__file__).parent.parent / "config" / "prompts" / "speculative_draft.txt"
).read_text()

# Mixture-of-Thought: Three diverse reasoning strategies
REASONING_STRATEGIES = [
    {
        "name": "step_by_step",
        "instruction": (
            "Think step-by-step. Break the question into logical steps and "
            "address each step using the evidence in the context. Show your reasoning chain."
        ),
    },
    {
        "name": "evidence_first",
        "instruction": (
            "Evidence-first approach. First identify ALL relevant evidence in the context, "
            "then synthesize the evidence into a comprehensive answer. "
            "Quote specific passages as evidence."
        ),
    },
    {
        "name": "contrarian_check",
        "instruction": (
            "Critical analysis approach. Consider what the answer might be, then look for "
            "evidence that CONTRADICTS or NUANCES that answer. Present a balanced view that "
            "accounts for conflicting information in the context."
        ),
    },
]


@dataclass
class Draft:
    strategy: str
    answer: str
    context_subset: list[str]  # doc_ids used
    confidence: float = 0.0


@dataclass
class SpeculativeResult:
    final_answer: str
    selected_draft_index: int
    drafts: list[Draft] = field(default_factory=list)
    verification_reasoning: str = ""
    citations: list[dict] = field(default_factory=list)


class SpeculativeRAG:
    """Generates parallel drafts with diverse reasoning, then verifies with strong model."""

    def __init__(
        self,
        draft_model: str = settings.draft_model,
        verifier_model: str = settings.verifier_model,
        num_drafts: int = settings.speculative_num_drafts,
    ):
        self.draft_model = draft_model
        self.verifier_model = verifier_model
        self.num_drafts = min(num_drafts, len(REASONING_STRATEGIES))

    def _partition_results(
        self, results: list[FusedResult], num_partitions: int
    ) -> list[list[FusedResult]]:
        """Partition results into disjoint subsets for diverse perspectives."""
        if not results:
            return [[] for _ in range(num_partitions)]

        # Strategy: distribute round-robin, ensuring each partition gets unique docs
        partitions: list[list[FusedResult]] = [[] for _ in range(num_partitions)]
        for i, result in enumerate(results):
            partitions[i % num_partitions].append(result)

        return partitions

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=15),
    )
    async def _generate_draft(
        self,
        query: str,
        context_results: list[FusedResult],
        strategy: dict,
    ) -> Draft:
        """Generate a single draft answer using a specific reasoning strategy."""
        context_text = "\n\n".join(f"[{r.doc_id}]: {r.text}" for r in context_results)

        prompt = (
            DRAFT_PROMPT.replace("{query}", query)
            .replace("{context}", context_text)
            .replace("{strategy}", strategy["name"])
            .replace("{strategy_instruction}", strategy["instruction"])
        )

        response = await litellm.acompletion(
            model=self.draft_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )

        return Draft(
            strategy=strategy["name"],
            answer=response.choices[0].message.content,
            context_subset=[r.doc_id for r in context_results],
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=20),
    )
    async def _verify_and_synthesize(
        self, query: str, drafts: list[Draft], all_results: list[FusedResult]
    ) -> SpeculativeResult:
        """Verifier model evaluates all drafts and synthesizes the best answer."""
        drafts_text = ""
        for i, draft in enumerate(drafts):
            drafts_text += f"\n--- DRAFT {i + 1} (Strategy: {draft.strategy}) ---\n{draft.answer}\n"

        context_text = "\n\n".join(f"[{r.doc_id}]: {r.text[:300]}" for r in all_results[:8])

        response = await litellm.acompletion(
            model=self.verifier_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a verification and synthesis expert. You are given multiple "
                        "draft answers to the same question, each using a different reasoning "
                        "strategy. Your job is to:\n"
                        "1. Evaluate which draft(s) are most accurate and well-supported\n"
                        "2. Synthesize the BEST final answer combining the strongest elements\n"
                        "3. Add inline citations [doc_id] for key claims\n"
                        "4. Return valid JSON with the structure shown below"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"## Question:\n{query}\n\n"
                        f"## Source Context:\n{context_text}\n\n"
                        f"## Draft Answers:\n{drafts_text}\n\n"
                        f"## Return valid JSON:\n"
                        f"{{\n"
                        f'  "selected_draft": <1-based index of best draft>,\n'
                        f'  "reasoning": "<why this draft is best>",\n'
                        f'  "final_answer": "<synthesized answer with [doc_id] citations>",\n'
                        f'  "citations": [{{"claim": "<claim>", "source": "<doc_id>"}}]\n'
                        f"}}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        return SpeculativeResult(
            final_answer=result.get("final_answer", drafts[0].answer),
            selected_draft_index=result.get("selected_draft", 1) - 1,
            drafts=drafts,
            verification_reasoning=result.get("reasoning", ""),
            citations=result.get("citations", []),
        )

    async def generate(
        self,
        query: str,
        results: list[FusedResult],
    ) -> SpeculativeResult:
        """
        Full Speculative RAG pipeline:
        1. Partition docs into N subsets
        2. Generate N drafts in parallel (Mixture-of-Thought)
        3. Verify and synthesize with strong model
        """
        # Step 1: Partition results
        partitions = self._partition_results(results, self.num_drafts)

        # Step 2: Generate drafts in parallel (Mixture-of-Thought)
        draft_tasks = []
        for i, (partition, strategy) in enumerate(
            zip(partitions, REASONING_STRATEGIES[: self.num_drafts])
        ):
            if not partition:
                # Use all results if partition is empty
                partition = results
            draft_tasks.append(self._generate_draft(query, partition, strategy))

        drafts = await asyncio.gather(*draft_tasks, return_exceptions=True)

        # Filter out failed drafts
        valid_drafts = [d for d in drafts if isinstance(d, Draft)]
        if not valid_drafts:
            logger.error("All drafts failed, generating basic response")
            return SpeculativeResult(
                final_answer="Unable to generate a response from the available context.",
                selected_draft_index=0,
            )

        logger.info(
            f"Generated {len(valid_drafts)}/{self.num_drafts} drafts "
            f"(strategies: {[d.strategy for d in valid_drafts]})"
        )

        # Step 3: Verify and synthesize
        result = await self._verify_and_synthesize(query, valid_drafts, results)

        logger.info(
            f"Verifier selected draft #{result.selected_draft_index + 1} "
            f"({valid_drafts[result.selected_draft_index].strategy if result.selected_draft_index < len(valid_drafts) else 'N/A'})"
        )
        return result

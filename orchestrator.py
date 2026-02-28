"""
Orchestrator — The main pipeline that wires all components together.
Routes: query -> security -> router -> [hyde] -> retrieval -> fusion ->
        rerank -> crag -> speculative_rag -> self_rag -> response
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from config.settings import RetrievalStrategy, settings
from generation.crag_evaluator import CRAGEvaluator, RetrievalQuality
from generation.response_builder import (
    PipelineResponse,
    ResponseBuilder,
    RetrievalTrace,
)
from generation.self_rag import SelfRAGReflector
from generation.speculative_rag import SpeculativeRAG
from ingestion.document_processor import DocumentChunk, DocumentProcessor
from ingestion.entity_extractor import EntityExtractor
from ingestion.graph_builder import GraphBuilder, KnowledgeGraph
from ingestion.tree_builder import RaptorTree, TreeBuilder
from ingestion.vector_indexer import VectorIndexer
from retrieval.graph_retriever import GraphRetriever
from retrieval.hybrid_fuser import FusedResult, HybridFuser
from retrieval.hyde import HyDEAugmenter
from retrieval.query_router import QueryRouter
from retrieval.reranker import ReRanker
from retrieval.tree_retriever import TreeRetriever
from security.access_control import AccessController, Role, User
from security.audit import AuditLogger
from security.pii_filter import PIIFilter
from security.prompt_guard import PromptGuard

logger = logging.getLogger(__name__)


class GraphTreeRAG:
    """
    Main orchestrator for the Graph-Tree Hybrid RAG pipeline.
    Combines 8 research papers into a unified, modular system.
    """

    def __init__(self):
        # Security
        self.prompt_guard = PromptGuard()
        self.pii_filter = PIIFilter()
        self.access_controller = AccessController()
        self.audit = AuditLogger()

        # Retrieval components (initialized on load/ingest)
        self.vector_indexer = VectorIndexer()
        self.kg: Optional[KnowledgeGraph] = None
        self.tree: Optional[RaptorTree] = None
        self.chunks: list[DocumentChunk] = []

        # Pipeline components
        self.router = QueryRouter()
        self.hyde = HyDEAugmenter()
        self.fuser = HybridFuser()
        self.crag = CRAGEvaluator()
        self.speculative = SpeculativeRAG()
        self.self_rag = SelfRAGReflector()
        self.response_builder = ResponseBuilder()

        # Lazy-initialized retrieval channels
        self._graph_retriever: Optional[GraphRetriever] = None
        self._tree_retriever: Optional[TreeRetriever] = None
        self._reranker: Optional[ReRanker] = None

    @property
    def graph_retriever(self) -> Optional[GraphRetriever]:
        if self._graph_retriever is None and self.kg and self.chunks:
            self._graph_retriever = GraphRetriever(self.kg, self.chunks)
        return self._graph_retriever

    @property
    def tree_retriever(self) -> Optional[TreeRetriever]:
        if self._tree_retriever is None and self.tree:
            self._tree_retriever = TreeRetriever(self.tree)
        return self._tree_retriever

    @property
    def reranker(self) -> ReRanker:
        if self._reranker is None:
            self._reranker = ReRanker(kg=self.kg)
        return self._reranker

    # ── Ingestion ──────────────────────────────────────────────────────

    async def ingest(
        self,
        source_path: Path,
        glob_pattern: str = "**/*.*",
    ) -> dict:
        """Full ingestion pipeline: chunk -> extract -> build graph/tree/index."""
        start = time.time()
        logger.info(f"Starting ingestion from {source_path}")

        # Step 1: Process documents into chunks
        processor = DocumentProcessor()
        if source_path.is_file():
            self.chunks = processor.process_file(source_path)
        else:
            self.chunks = processor.process_directory(source_path, glob_pattern)
        logger.info(f"Processed {len(self.chunks)} chunks")

        # Step 2: Extract entities and relationships
        extractor = EntityExtractor()
        entities, relationships = await extractor.extract_from_chunks(self.chunks)

        # Step 3: Build knowledge graph with communities
        graph_builder = GraphBuilder()
        self.kg = graph_builder.build(entities, relationships, self.chunks)
        await graph_builder.generate_all_summaries(self.kg, self.chunks)
        graph_builder.save(self.kg)

        # Step 4: Build RAPTOR tree
        tree_builder = TreeBuilder()
        self.tree = await tree_builder.build(self.chunks)
        tree_builder.save(self.tree)

        # Step 5: Build vector + BM25 indices
        self.vector_indexer.build_from_chunks(self.chunks, self.tree)
        self.vector_indexer.save()

        # Reset lazy-loaded components
        self._graph_retriever = None
        self._tree_retriever = None
        self._reranker = None

        elapsed = time.time() - start
        stats = {
            "chunks": len(self.chunks),
            "entities": len(entities),
            "relationships": len(relationships),
            "graph_nodes": self.kg.graph.number_of_nodes(),
            "graph_edges": self.kg.graph.number_of_edges(),
            "communities": len(self.kg.communities),
            "tree_nodes": len(self.tree.nodes),
            "tree_levels": self.tree.max_level + 1,
            "vector_index_size": self.vector_indexer.faiss_index.ntotal
            if self.vector_indexer.faiss_index
            else 0,
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info(f"Ingestion complete: {stats}")
        return stats

    def load(self) -> None:
        """Load pre-built indices from disk."""
        graph_builder = GraphBuilder()
        tree_builder = TreeBuilder()

        if settings.graph_persist_path.exists():
            self.kg = graph_builder.load()
        if settings.tree_persist_path.exists():
            self.tree = tree_builder.load()
        if settings.faiss_index_path.exists():
            self.vector_indexer.load()

        # Reset lazy-loaded components
        self._graph_retriever = None
        self._tree_retriever = None
        self._reranker = None

        logger.info("Loaded all indices from disk")

    # ── Query Pipeline ─────────────────────────────────────────────────

    async def query(
        self,
        query: str,
        user: Optional[User] = None,
    ) -> PipelineResponse:
        """
        Full query pipeline:
        1. Security gate (prompt guard + PII filter)
        2. Adaptive routing
        3. HyDE augmentation (optional)
        4. 4-channel parallel retrieval
        5. Hybrid fusion + re-ranking
        6. CRAG evaluation + corrective action
        7. Speculative RAG (Mixture-of-Thought)
        8. Self-RAG reflection + correction
        9. Response assembly with provenance
        """
        start = time.time()
        trace_id = self.audit.generate_trace_id()
        user = user or User(user_id="anonymous", role=Role.USER)

        trace = RetrievalTrace()

        # ── Step 1: Security Gate ──────────────────────────────────────
        guard_result = self.prompt_guard.check(query)
        if not guard_result.is_safe:
            self.audit.log_security_event(
                trace_id,
                "injection_blocked",
                {"reason": guard_result.blocked_reason, "risk": guard_result.risk_score},
                user.user_id,
            )
            return PipelineResponse(
                query=query,
                answer="Query blocked for security reasons.",
                confidence=0.0,
                trace_id=trace_id,
                warnings=[guard_result.blocked_reason or "Injection detected"],
            )

        clean_query = guard_result.sanitized_query or query
        pii_result = self.pii_filter.filter_text(clean_query)
        clean_query = pii_result.filtered_text

        self.audit.log_query(trace_id, clean_query, user.user_id, user.role.value)

        # ── Step 2: Adaptive Routing ───────────────────────────────────
        plan = await self.router.route(clean_query)
        trace.router_strategy = plan.strategy.value
        trace.router_query_type = plan.query_type.value
        trace.sub_questions = [sq.question for sq in plan.sub_questions]

        # Simple queries: direct LLM response (no retrieval)
        if plan.strategy == RetrievalStrategy.SIMPLE:
            import litellm

            response = await litellm.acompletion(
                model=settings.strong_model,
                messages=[{"role": "user", "content": clean_query}],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            elapsed_ms = (time.time() - start) * 1000
            trace.total_latency_ms = elapsed_ms
            return self.response_builder.build(
                query=query,
                answer=response.choices[0].message.content,
                context_results=[],
                trace=trace,
                trace_id=trace_id,
            )

        # ── Step 3: HyDE Augmentation ─────────────────────────────────
        hyde_result = None
        if plan.hyde_enabled and settings.hyde_enabled:
            hyde_result = await self.hyde.augment(clean_query)
            trace.hyde_used = True
            trace.hyde_document = hyde_result.hypothetical_document[:200]

        # ── Step 4: 4-Channel Parallel Retrieval ──────────────────────
        query_embedding = self.vector_indexer.embed_query(clean_query)

        # Run retrieval channels in parallel
        retrieval_tasks = {}

        # Vector + HyDE
        hyde_emb = hyde_result.hyde_embedding if hyde_result else None
        vector_results = self.vector_indexer.search_hybrid_embedding(query_embedding, hyde_emb)
        retrieval_tasks["vector"] = vector_results

        # BM25
        bm25_results = self.vector_indexer.search_sparse(clean_query)
        retrieval_tasks["bm25"] = bm25_results

        # Graph
        graph_results_raw = []
        if self.graph_retriever:
            graph_results_raw = await self.graph_retriever.retrieve(clean_query)
        graph_results = [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                "source_type": r.source_type,
                "entities_matched": r.entities_matched,
                "metadata": r.metadata,
            }
            for r in graph_results_raw
        ]
        retrieval_tasks["graph"] = graph_results

        # Tree
        tree_results_raw = []
        if self.tree_retriever:
            tree_results_raw = self.tree_retriever.retrieve(query_embedding)
        tree_results = [
            {
                "node_id": r.node_id,
                "text": r.text,
                "score": r.score,
                "source_type": r.source_type,
                "metadata": r.metadata,
            }
            for r in tree_results_raw
        ]
        retrieval_tasks["tree"] = tree_results

        trace.channels_used = list(retrieval_tasks.keys())
        trace.channel_result_counts = {k: len(v) for k, v in retrieval_tasks.items()}

        # ── Step 5: Hybrid Fusion + Re-ranking ────────────────────────
        fused_results = self.fuser.fuse(
            vector_results=retrieval_tasks["vector"],
            bm25_results=retrieval_tasks["bm25"],
            graph_results=retrieval_tasks["graph"],
            tree_results=retrieval_tasks["tree"],
        )

        # RBAC filtering
        fused_results = self.access_controller.filter_results(user, fused_results)

        # Re-ranking
        query_entities = [e for r in graph_results_raw for e in r.entities_matched]
        reranked = self.reranker.rerank(clean_query, fused_results, query_entities=query_entities)

        self.audit.log_retrieval(
            trace_id,
            plan.strategy.value,
            len(reranked),
            trace.channel_result_counts,
        )

        # ── Step 6: CRAG Evaluation ───────────────────────────────────
        crag_eval = None
        context_results = reranked
        if settings.crag_enabled:
            crag_eval = await self.crag.evaluate(clean_query, reranked)
            trace.crag_quality = crag_eval.quality.value
            trace.crag_action = crag_eval.recommended_action

            if crag_eval.quality == RetrievalQuality.INCORRECT and crag_eval.web_results:
                # Convert web results to FusedResult format
                for wr in crag_eval.web_results:
                    context_results.append(
                        FusedResult(
                            doc_id=f"web_{hash(wr.get('url', ''))}"[:16],
                            text=wr.get("text", ""),
                            fused_score=wr.get("score", 0.5),
                            source_type="web_search",
                            metadata={"url": wr.get("url", ""), "title": wr.get("title", "")},
                        )
                    )
                trace.web_search_used = True
            elif crag_eval.filtered_results:
                context_results = crag_eval.filtered_results

        # ── Step 7: Speculative RAG (Mixture-of-Thought) ──────────────
        if settings.speculative_enabled and context_results:
            spec_result = await self.speculative.generate(clean_query, context_results)
            answer = spec_result.final_answer
            trace.speculative_strategies = [d.strategy for d in spec_result.drafts]
            trace.speculative_selected = (
                spec_result.drafts[spec_result.selected_draft_index].strategy
                if spec_result.selected_draft_index < len(spec_result.drafts)
                else ""
            )
        else:
            # Fallback: direct generation
            import litellm

            context_text = "\n\n".join(r.text[:500] for r in context_results[:8])
            response = await litellm.acompletion(
                model=settings.strong_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Answer using ONLY the provided context. Cite sources with [doc_id].",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context_text}\n\nQuestion: {clean_query}",
                    },
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            answer = response.choices[0].message.content

        # ── Step 8: Self-RAG Reflection ───────────────────────────────
        reflection_scores = {}
        if settings.self_rag_enabled and context_results:
            self_rag_result = await self.self_rag.reflect_and_correct(
                clean_query, answer, context_results
            )
            answer = self_rag_result.final_answer
            trace.self_rag_iterations = self_rag_result.iterations
            reflection_scores = {
                "relevance": self_rag_result.reflection.relevance,
                "support": self_rag_result.reflection.support,
                "usefulness": self_rag_result.reflection.usefulness,
                "citations": self_rag_result.reflection.citations,
            }
            trace.self_rag_scores = reflection_scores

        # ── Step 9: PII filter on output ──────────────────────────────
        output_pii = self.pii_filter.filter_text(answer)
        answer = output_pii.filtered_text

        # ── Step 10: Build Response ────────────────────────────────────
        elapsed_ms = (time.time() - start) * 1000
        trace.total_latency_ms = elapsed_ms

        response = self.response_builder.build(
            query=query,
            answer=answer,
            context_results=context_results,
            trace=trace,
            reflection_scores=reflection_scores,
            trace_id=trace_id,
        )

        self.audit.log_response(
            trace_id, query, answer, response.confidence, elapsed_ms, len(response.sources)
        )

        self.audit.log_generation(
            trace_id,
            model=settings.strong_model,
            strategy="speculative" if settings.speculative_enabled else "direct",
            num_drafts=len(trace.speculative_strategies),
            self_rag_iterations=trace.self_rag_iterations,
            confidence=response.confidence,
            latency_ms=elapsed_ms,
        )

        return response

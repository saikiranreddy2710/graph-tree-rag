"""
FastAPI Server — REST API for the Graph-Tree Hybrid RAG pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from graph_tree_rag import __version__
from orchestrator import GraphTreeRAG
from security.access_control import Role, User

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Graph-Tree Hybrid RAG",
    version=__version__,
    description=(
        "Advanced Hybrid RAG combining tree-structured routing, graph traversal, "
        "speculative generation, and self-reflective correction. "
        "Based on 8 research papers (GraphRAG, RAPTOR, CRAG, Self-RAG, "
        "Speculative RAG, HyDE, Adaptive-RAG, Modular RAG)."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = GraphTreeRAG()


@app.on_event("startup")
async def startup():
    """Load indices on startup if available."""
    try:
        pipeline.load()
        logger.info("Pipeline loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load indices on startup: {e}")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version=__version__,
        indices_loaded=pipeline.vector_indexer.faiss_index is not None,
        graph_nodes=pipeline.kg.graph.number_of_nodes() if pipeline.kg else 0,
        tree_nodes=len(pipeline.tree.nodes) if pipeline.tree else 0,
        vector_count=(
            pipeline.vector_indexer.faiss_index.ntotal if pipeline.vector_indexer.faiss_index else 0
        ),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run the full hybrid RAG pipeline."""
    user = User(
        user_id=request.user_id,
        role=Role(request.user_role) if request.user_role in Role.__members__ else Role.USER,
    )

    try:
        response = await pipeline.query(request.query, user=user)
        return QueryResponse(**response.to_dict())
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest documents into the pipeline."""
    source = Path(request.source_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {source}")

    try:
        stats = await pipeline.ingest(source, request.glob_pattern)
        return IngestResponse(status="success", stats=stats)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/{trace_id}")
async def explain(trace_id: str):
    """Get retrieval explanation for a past query by trace_id."""
    # Read from audit log
    import json
    from config.settings import settings

    entries = []
    try:
        with open(settings.audit_log_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("trace_id") == trace_id:
                    entries.append(entry)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No audit log found")

    if not entries:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    return {"trace_id": trace_id, "events": entries}

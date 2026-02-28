<div align="center">

# Graph-Tree RAG

### A Hybrid Retrieval-Augmented Generation System

**Tree Routing + Graph Traversal + Speculative Generation + Self-Correction**

Eight recent research papers. One unified pipeline.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-42%20passed-brightgreen.svg)](#testing)
[![arXiv Papers](https://img.shields.io/badge/arXiv-8%20papers-b31b1b.svg)](#research-foundation)

---

[Why This Exists](#why-this-exists) &bull;
[How It Works](#how-it-works) &bull;
[Quick Start](#quick-start) &bull;
[Architecture](#architecture) &bull;
[Research Foundation](#research-foundation) &bull;
[Evaluation](#evaluation)

</div>

---

## Why This Exists

Standard RAG retrieves the top-K most similar chunks to a query and feeds them to an LLM. That works for simple lookups. It falls apart when:

- The answer requires **connecting facts across multiple documents** (multi-hop reasoning).
- The query is **global** ("What are the main themes in this dataset?") rather than a point lookup.
- Two documents are **semantically similar but factually different** -- dense retrieval can't tell them apart.
- The retrieved context is **noisy, incomplete, or outright wrong**, and the LLM hallucinates anyway.

Graph-Tree RAG addresses all four problems by combining structured retrieval (knowledge graphs, summarization trees) with dense/sparse search, and then verifying its own output before returning an answer.

This is not a wrapper around someone else's library. Every component -- from entity extraction to community detection to speculative draft generation -- is implemented here, grounded in specific published research, and wired into a single pipeline.

---

## How It Works

A query goes through ten steps. Each one exists for a reason.

```
Query
  |
  v
[1] Security Gate ........... Block prompt injections. Redact PII from input.
  |
  v
[2] Adaptive Router ......... Classify complexity. Simple queries skip retrieval entirely.
  |                           Complex queries get decomposed into sub-questions (Chain-of-Thought).
  v
[3] HyDE Augmentation ....... Generate a hypothetical answer. Embed it. Use both the
  |                           query embedding and the hypothetical embedding to widen recall.
  v
[4] Four-Channel Retrieval .. Run four searches in parallel:
  |                             - Dense vector search (FAISS)
  |                             - Sparse keyword search (BM25)
  |                             - Knowledge graph walk (entity-centric, up to 2 hops)
  |                             - RAPTOR tree search (all abstraction levels at once)
  v
[5] Hybrid Fusion ........... Merge results via Reciprocal Rank Fusion with learned
  |                           per-channel weights. A document found by multiple channels
  |                           gets a stronger signal than one found by a single channel.
  v
[6] Re-Ranking .............. Cross-encoder fine-grained scoring. Graph-coherence bonus
  |                           for chunks whose entities connect to query entities in the KG.
  |                           MMR diversity filter to avoid near-duplicate context.
  v
[7] CRAG Evaluation ......... LLM judges whether the retrieved context actually answers
  |                           the query. Three outcomes:
  |                             CORRECT   -> proceed
  |                             AMBIGUOUS -> decompose query, re-retrieve
  |                             INCORRECT -> fall back to web search (Tavily)
  v
[8] Speculative Generation .. Partition context into 3 subsets. Generate 3 draft answers
  |                           in parallel, each with a different reasoning strategy
  |                           (step-by-step, evidence-first, contrarian-check).
  |                           A stronger verifier model picks the best draft and synthesizes.
  v
[9] Self-RAG Reflection ..... Evaluate the answer on four axes: relevance, factual support,
  |                           usefulness, citation accuracy. If any score is below threshold,
  |                           regenerate with feedback. Up to 2 correction loops.
  v
[10] Response ............... Final answer with inline citations, source provenance,
                              confidence score, and a full retrieval trace explaining
                              every decision the pipeline made.
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- At least one LLM API key (OpenAI or Anthropic)

### Installation

```bash
git clone https://github.com/saikiranreddy2710/graph-tree-rag.git
cd graph-tree-rag

# Create virtual environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
```

Open `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-...
# and/or
ANTHROPIC_API_KEY=sk-ant-...

# Optional: web search fallback
GTR_WEB_SEARCH_API_KEY=tvly-...
```

### Ingest Documents

```bash
# Ingest a directory of documents (txt, md, html, json, py, etc.)
python cli.py ingest /path/to/your/documents

# Ingest a single file
python cli.py ingest /path/to/paper.md
```

This runs the full ingestion pipeline: chunking, entity extraction, knowledge graph construction (with Leiden community detection and community summaries), RAPTOR tree building, and FAISS/BM25 indexing. Progress is logged to the console.

### Query

```bash
# Basic query
python cli.py query "What are the main approaches to retrieval-augmented generation?"

# Verbose mode -- shows retrieval trace, sources, and pipeline decisions
python cli.py query "Compare GraphRAG and RAPTOR" -v

# JSON output for programmatic use
python cli.py query "Why does vector search fail on overlapping topics?" -j
```

### Start the API Server

```bash
python cli.py serve --port 8000
```

Then query via HTTP:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Leiden algorithm?"}'
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Run the full hybrid RAG pipeline |
| `POST` | `/ingest` | Ingest documents from a path |
| `GET` | `/health` | System status and index stats |
| `GET` | `/explain/{trace_id}` | Retrieval explanation for a past query |

---

## Architecture

### Project Structure

```
graph_tree_rag/
├── config/
│   ├── settings.py                  # All configuration (env-driven, Pydantic)
│   └── prompts/                     # 7 versioned prompt templates
│       ├── entity_extraction.txt
│       ├── community_summary.txt
│       ├── hyde_generation.txt
│       ├── query_decomposition.txt
│       ├── speculative_draft.txt
│       ├── self_rag_reflection.txt
│       └── crag_evaluation.txt
│
├── ingestion/                       # Document processing & index building
│   ├── document_processor.py        # Semantic chunking with overlap
│   ├── entity_extractor.py          # LLM-based NER with confidence scores
│   ├── graph_builder.py             # KG construction + Leiden communities
│   ├── tree_builder.py              # RAPTOR bottom-up summarization tree
│   └── vector_indexer.py            # FAISS dense + BM25 sparse indices
│
├── retrieval/                       # Query routing & multi-channel retrieval
│   ├── query_router.py              # Adaptive 3-way complexity routing
│   ├── hyde.py                      # Hypothetical Document Embeddings
│   ├── graph_retriever.py           # Entity walk + community expansion
│   ├── tree_retriever.py            # Collapsed-tree multi-level search
│   ├── hybrid_fuser.py              # 4-channel Reciprocal Rank Fusion
│   └── reranker.py                  # Cross-encoder + MMR + graph coherence
│
├── generation/                      # Answer generation & verification
│   ├── crag_evaluator.py            # Retrieval quality gate + web fallback
│   ├── speculative_rag.py           # 3 parallel drafts (Mixture-of-Thought)
│   ├── self_rag.py                  # Reflection scoring + correction loops
│   └── response_builder.py          # Final assembly with provenance
│
├── security/                        # Production security layer
│   ├── prompt_guard.py              # Prompt injection detection
│   ├── pii_filter.py                # PII detection & redaction
│   ├── access_control.py            # Role-based access control (RBAC)
│   └── audit.py                     # Structured JSONL audit logging
│
├── evaluation/                      # Quality measurement & regression
│   ├── metrics.py                   # P@K, MRR, NDCG, entity coverage, etc.
│   ├── regression_suite.py          # Baseline comparison with fail gates
│   ├── ablation_runner.py           # 9-config comparative evaluation
│   ├── failure_analyzer.py          # Auto-categorize + recommendations
│   └── test_queries/                # Curated gold-standard queries
│
├── observability/                   # Tracing & explainability
│   ├── tracer.py                    # OpenTelemetry integration
│   └── explainer.py                 # Human-readable retrieval explanations
│
├── api/
│   ├── server.py                    # FastAPI REST endpoints
│   └── schemas.py                   # Request/Response Pydantic models
│
├── orchestrator.py                  # Main pipeline wiring (10 steps)
├── cli.py                           # Typer CLI (ingest, query, serve, evaluate)
└── tests/                           # 42 unit tests
```

### Key Design Decisions

**Why four retrieval channels instead of one?**
Dense vector search finds semantically similar content. BM25 finds exact keyword matches that embeddings sometimes miss. Graph traversal finds causally or structurally connected information that is relevant but not similar. Tree search finds information at different levels of abstraction. Each channel catches things the others miss. Reciprocal Rank Fusion combines them without needing to normalize scores across different similarity spaces.

**Why Speculative RAG with three drafts?**
A single LLM call sees all the context at once and can develop position bias (over-relying on documents that appear first). By partitioning context into three subsets and using three different reasoning strategies (step-by-step, evidence-first, contrarian-check), we get diverse perspectives. The verifier then has multiple candidates to choose from and can synthesize the strongest elements of each.

**Why two correction loops (CRAG + Self-RAG)?**
CRAG catches retrieval failures early -- before generation starts. If the context is wrong, no amount of generation quality will save the answer. Self-RAG catches generation failures after the fact -- unsupported claims, missing citations, incomplete answers. They operate at different points in the pipeline and catch different failure modes.

**Why not use LangChain or LlamaIndex?**
This project implements specific research papers with full control over every component. Framework abstractions would obscure the pipeline mechanics and limit the ability to tune each stage independently. Every module here is a plain Python class with clear inputs, outputs, and no hidden magic.

---

## Research Foundation

Every major component maps to a specific published paper. This is not a collection of buzzwords -- each technique was chosen because it solves a concrete retrieval or generation failure mode.

| Paper | Venue | What We Use From It |
|-------|-------|---------------------|
| [**GraphRAG**](https://arxiv.org/abs/2404.16130) -- Edge et al. (Microsoft) | arXiv 2024 | Entity knowledge graph, Leiden community detection, hierarchical community summaries for global queries |
| [**RAPTOR**](https://arxiv.org/abs/2401.18059) -- Sarthi et al. (Stanford) | arXiv 2024 | Recursive bottom-up clustering and abstractive summarization tree, collapsed-tree retrieval across abstraction levels |
| [**Speculative RAG**](https://arxiv.org/abs/2407.08223) -- Wang et al. (Google) | ICLR 2025 | Parallel draft generation from document subsets with a smaller model, single-pass verification with a stronger model |
| [**Self-RAG**](https://arxiv.org/abs/2310.11511) -- Asai et al. | arXiv 2023 | Reflection tokens for relevance, support, and usefulness assessment; on-demand corrective retrieval loops |
| [**CRAG**](https://arxiv.org/abs/2401.15884) -- Yan et al. | arXiv 2024 | Retrieval quality evaluator with confidence-gated actions (proceed / decompose / web search fallback) |
| [**Adaptive-RAG**](https://arxiv.org/abs/2403.14403) -- Jeong et al. | NAACL 2024 | Query complexity classifier that routes simple, moderate, and complex queries to different retrieval strategies |
| [**HyDE**](https://arxiv.org/abs/2212.10496) -- Gao et al. | arXiv 2022 | Hypothetical document generation for zero-shot dense retrieval; the generated doc captures relevance patterns even when the query is vague |
| [**Modular RAG**](https://arxiv.org/abs/2407.21059) -- Gao et al. | arXiv 2024 | LEGO-like reconfigurable pipeline design with routing, scheduling, and fusion mechanisms; conditional and branching flow patterns |

---

## Configuration

All settings are driven by environment variables (prefixed with `GTR_`) or a `.env` file. Every feature can be toggled independently.

```env
# Models
GTR_FAST_MODEL=gpt-4o-mini               # Extraction, routing, drafting
GTR_STRONG_MODEL=claude-3-5-sonnet-20241022  # Verification, final generation

# Feature Flags
GTR_HYDE_ENABLED=true                     # Hypothetical document embeddings
GTR_CRAG_ENABLED=true                     # Corrective retrieval evaluation
GTR_SPECULATIVE_ENABLED=true              # Mixture-of-Thought parallel drafts
GTR_SELF_RAG_ENABLED=true                 # Self-reflective correction loops
GTR_WEB_SEARCH_ENABLED=true               # Tavily web search fallback

# Retrieval Tuning
GTR_FUSION_WEIGHTS_VECTOR=0.30            # Dense search weight in RRF
GTR_FUSION_WEIGHTS_BM25=0.15              # Sparse search weight
GTR_FUSION_WEIGHTS_GRAPH=0.30             # Graph traversal weight
GTR_FUSION_WEIGHTS_TREE=0.25              # Tree retrieval weight

# Security
GTR_SECURITY_PROMPT_GUARD_ENABLED=true
GTR_SECURITY_PII_FILTER_ENABLED=true
GTR_SECURITY_RBAC_ENABLED=true
```

See [`config/settings.py`](config/settings.py) for the full list of ~50 configurable parameters.

---

## Evaluation

### Built-In Metrics

The evaluation harness computes retrieval and generation metrics without external dependencies:

**Retrieval**: Precision@K, Recall@K, MRR, NDCG@K, Hit Rate, Entity Coverage

**Generation**: Faithfulness (claim support), Answer Relevance, Context Precision, Citation Accuracy, Answer Similarity (vs gold)

### Regression Testing

```bash
python cli.py evaluate
```

Compares current metrics against a saved baseline. Any metric that drops below a configurable threshold fails the suite. This catches regressions from prompt changes, retrieval tuning, or model swaps.

### Ablation Study

The ablation runner tests 9 configurations to quantify each component's contribution:

| Configuration | What It Tests |
|---------------|---------------|
| `vector_only` | Dense retrieval alone (the baseline most RAG systems use) |
| `vector_bm25` | Dense + sparse (hybrid search without structure) |
| `graph_only` | Knowledge graph traversal alone |
| `tree_only` | RAPTOR tree retrieval alone |
| `hybrid_no_hyde` | Full 4-channel without HyDE augmentation |
| `hybrid_no_crag` | Full hybrid without corrective retrieval |
| `hybrid_no_speculative` | Full hybrid with direct generation instead of MoT drafts |
| `hybrid_no_self_rag` | Full hybrid without reflection/correction loops |
| `full_hybrid` | Everything enabled (the complete pipeline) |

### Failure Analysis

When a query produces a bad answer, the failure analyzer categorizes it automatically:

- **WRONG_ENTITY** -- Retrieved docs about a different entity with a similar name
- **MISSING_CONTEXT** -- The answer exists in the corpus but wasn't retrieved
- **HALLUCINATION** -- The answer contains claims not supported by any retrieved context
- **STALE_DATA** -- Retrieved information is outdated
- **NO_RETRIEVAL** -- No results were found at all
- **WRONG_REASONING** -- Context was correct but the LLM reasoned incorrectly

Each failure type comes with specific recommendations for what to tune.

---

## Security

This is not an afterthought. The security layer runs on every query, both input and output.

### Prompt Injection Detection

Pattern-based detection for known injection attacks: instruction override ("ignore previous instructions"), system prompt extraction ("print your system prompt"), role-play jailbreaks ("you are now DAN"), delimiter injection (`[INST]`, `<<SYS>>`). Queries that exceed a risk threshold are blocked outright. Lower-risk queries are sanitized.

### PII Filtering

Regex-based detection and redaction of emails, phone numbers, SSNs, credit card numbers, IP addresses, API keys, and AWS credentials. Runs on both the user's input query and the generated output. Optionally integrates with Microsoft Presidio for NER-based detection.

### Role-Based Access Control

Documents can be tagged with access roles (`admin`, `user`, `restricted`, `guest`). At retrieval time, results are filtered so users only see documents their role permits. This is enforced at the retrieval layer, not the application layer.

### Audit Logging

Every query, retrieval, generation, and security event is logged as structured JSON with a correlation trace ID. The audit log at `data/audit.jsonl` provides a complete forensic trail.

---

## Testing

```bash
# Run all tests
PYTHONPATH=. pytest tests/ -v

# Run specific test suites
PYTHONPATH=. pytest tests/test_security.py -v      # Prompt injection & PII
PYTHONPATH=. pytest tests/test_metrics.py -v        # Evaluation metrics
PYTHONPATH=. pytest tests/test_query_router.py -v   # Query classification
PYTHONPATH=. pytest tests/test_hybrid_fuser.py -v   # RRF fusion logic
PYTHONPATH=. pytest tests/test_document_processor.py -v  # Chunking
```

Current: **42 tests passing** across 5 test modules covering security, metrics, routing, fusion, and document processing. Tests that require LLM API calls are designed to be run with live keys.

---

## Dependencies

Core dependencies are chosen for production reliability and minimal footprint:

| Category | Package | Purpose |
|----------|---------|---------|
| LLM | `litellm` | Unified API across OpenAI, Anthropic, and 100+ providers |
| Embeddings | `sentence-transformers` | Local embedding models (no API calls for embeddings) |
| Vector Search | `faiss-cpu` | Facebook's similarity search library |
| Sparse Search | `rank-bm25` | BM25 keyword retrieval |
| Graph | `networkx` + `graspologic` | Knowledge graph + Leiden community detection |
| Clustering | `scikit-learn` + `umap-learn` | RAPTOR tree construction (GMM + UMAP) |
| Re-ranking | `transformers` | Cross-encoder models for fine-grained scoring |
| API | `fastapi` + `uvicorn` | REST API server |
| CLI | `typer` + `rich` | Command-line interface with formatted output |
| Security | `presidio-analyzer` | PII detection (optional, regex fallback included) |
| Observability | `opentelemetry-sdk` | Distributed tracing |
| Resilience | `tenacity` | Retry logic for all LLM calls |

---

## What This Is Not

- **Not a framework.** This is a complete, opinionated pipeline. It doesn't try to be all things to all people.
- **Not a demo.** The security layer, evaluation harness, and failure analysis are production concerns. They are here because retrieval quality matters.
- **Not finished.** The ablation framework exists specifically to identify what works and what doesn't, so the system can improve iteratively.

---

## Acknowledgments

This project builds directly on the work of researchers at Microsoft, Stanford, Google, and several university labs. The papers are cited above and in the source code docstrings. The goal was not to repackage their work, but to combine their individually published ideas into a system that is greater than the sum of its parts.

---

<div align="center">

**Built by [Sai Kiran Reddy](https://github.com/saikiranreddy2710)**

If this is useful, a star helps others find it.

</div>

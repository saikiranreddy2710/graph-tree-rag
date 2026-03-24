<div align="center">

# Graph-Tree RAG

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

**Advanced Hybrid Retrieval-Augmented Generation (RAG) System**
*Tree Routing + Graph Traversal + Speculative Generation + Self-Correction*

[Features](#why-this-project-is-useful) • [Installation](#how-users-can-get-started) • [Usage](#usage) • [Help & Support](#where-users-can-get-help) • [Contributing](#who-maintains-and-contributes)

</div>

---

## What the Project Does

**Graph-Tree RAG** is a production-ready, highly modular Retrieval-Augmented Generation pipeline. It solves the limitations of standard vector-search RAG by connecting facts across multiple documents, answering global queries ("What are the main themes?"), and verifying its own generated output.

This project combines state-of-the-art research (GraphRAG, RAPTOR, CRAG, Self-RAG, MoT, HyDE) into a unified application. It implements a fully controllable, multi-channel retrieval process that includes dense vector search, sparse keyword search, knowledge graph traversal, and hierarchical tree search—alongside a novel, brain-inspired cortex storage module.

---

## Why the Project is Useful

Standard dense retrieval often fails on complex, multi-hop reasoning questions or overlapping semantics. Graph-Tree RAG is designed to overcome these hurdles with the following key features:

### ✨ Key Features and Benefits

- **Multi-Channel Retrieval**: Runs 4+ retrieval strategies concurrently (FAISS, BM25, KG entity walk, RAPTOR summaries, Cortex activation), merging results through learned Reciprocal Rank Fusion.
- **Adaptive Routing**: Intelligently skips retrieval for simple queries or decomposes complex queries using Chain-of-Thought (CoT) reasoning.
- **Self-Correction (Self-RAG & CRAG)**: Evaluates the retrieval context before generating. If the context is poor, it falls back to a web search. The pipeline then reflects on its own answer (relevance, factual support) and triggers correction loops if necessary.
- **Speculative Draft Generation (MoT)**: Partitions context and generates 3 diverse answer drafts in parallel using smaller models, synthesizing the best into the final response with a stronger model.
- **Built-In Security Layer**: Protects against prompt injections and immediately filters out PII data (using Presidio or Regex fallback) on both input and output. Role-based access control (RBAC) securely filters contexts at the retrieval level.

---

## How Users Can Get Started

### Prerequisites

- Python 3.11 or higher
- API Keys for OpenAI or Anthropic (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)

### Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/saikiranreddy2710/graph-tree-rag.git
   cd graph-tree-rag
   ```

2. **Set up the environment:**

   Create a virtual environment and install the package (with development dependencies).

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Configure the application:**

   Copy the example environment variables and add your own API credentials.

   ```bash
   cp .env.example .env
   # Edit .env file to include OPENAI_API_KEY or ANTHROPIC_API_KEY
   ```

### Usage Examples

#### 1. Ingest Documents

Before answering questions, ingest a set of documents to build the core indices (graph, tree, and vectors).

```bash
# Ingest an entire directory (supports .txt, .md, .html, .py, etc.)
gtr ingest /path/to/your/documents
```

#### 2. Query the Knowledge Base

You can interact with the pipeline using the command-line interface. Use the `--verbose` flag to see the exact retrieval trace, trace metrics, and pipeline decisions.

```bash
# Simple descriptive question
gtr query "What are the main approaches to retrieval-augmented generation?"

# Complex comparison, printing the full trace
gtr query "Compare GraphRAG and RAPTOR methodologies" --verbose
```

#### 3. Start the API Server

Serve the application as a standalone REST container using FastAPI:

```bash
gtr serve --port 8000
```

You can then issue `POST` requests directly via curl or view the interactive Swagger UI at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the concept of speculative generation."}'
```

---

## Where Users Can Get Help

If you encounter issues, have questions, or need guidance on tuning parameters for your specific data:

- **Issues / Bug Tracker**: Use the [GitHub Issues](https://github.com/saikiranreddy2710/graph-tree-rag/issues) page to report unexpected behavior or request new features.
- **Detailed Documentation**: For deep dives into the pipeline's structure, models config, and paper references, see our [docs/](docs/) folder (coming soon). Check the source code docstrings for low-level function references.
- **Evaluation Guide**: Run the built-in testing pipeline using `gtr evaluate` to track your evaluation metrics after tweaking parameters.

---

## Who Maintains and Contributes

This project is authored and maintained by **[Sai Kiran Reddy](https://github.com/saikiranreddy2710)**. 

### Contribution Guidelines

We welcome community contributions, ranging from adding new retrieval mechanisms or LLM providers to enhancing the test suite. 

To contribute:
1. Fork the repo and create your feature branch.
2. Ensure you add or update relevant unit tests in the `tests/` directory.
3. Verify the pipeline by running `pytest` (we ensure zero regressions).
4. For detailed coding standards, environment setup, and pull-request procedures, please refer to our [CONTRIBUTING.md](CONTRIBUTING.md).

### License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

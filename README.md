<div align="center">

# Graph-Tree RAG

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

**Advanced Brain-Inspired Retrieval-Augmented Generation (RAG) System**
*CortexStore + Tree Routing + Graph Traversal + Speculative Generation + Self-Correction*

[Features](#why-this-project-is-useful-our-new-goal) • [Architecture](#-the-cortexstore-architecture) • [Installation](#how-users-can-get-started) • [Usage](#usage-examples) • [Help & Support](#where-users-can-get-help)

</div>

---

## What the Project Does

**Graph-Tree RAG** is a production-ready, highly modular Retrieval-Augmented Generation pipeline. We have officially shifted our core architecture away from static neural graphs to a **dynamic, self-organizing storage model known as CortexStore**.

By integrating core cognitive mechanisms into the orchestrator pipeline, this system mimics the brain's nervous system behavior. Graph-Tree RAG gets smarter with every query. It solves the limitations of standard vector-search RAG by connecting facts across multiple documents, tracking its own usage patterns, and physically evolving its retrieval structure based on real user queries.

---

## Why the Project is Useful: Our New Goal

Our ultimate objective is to achieve **high-performance retrieval at scale (1M+ nodes)** while enabling the system to organically evolve and improve its accuracy. Standard dense retrieval fails on complex reasoning or overlapping semantics; standard graph traversal is static. We combine the best of both into a highly dynamic, living system.

### 🧠 The CortexStore Architecture
The new CortexStore models a biological brain through four profound cognitive mechanisms:

- **Hebbian Learning ("Neurons that fire together, wire together")**: Synaptic connections between chunks of knowledge learn from usage. When related facts are retrieved together, their connection weight strengthens automatically.
- **Synaptic Pruning & Memory Consolidation**: Like the brain during sleep, unused connections weaken and eventually decay. Highly similar knowledge nodes merge into consolidated concepts, optimizing storage without losing context.
- **Cortical Columns**: Knowledge self-organizes into semantic clusters (columns), allowing the retrieval pipeline to perform fast, macro-level activation before diving into individual nodes.
- **Working Memory Buffer**: A capacity-limited buffer tracks transient activations per query session, providing immediate context boosts for complex, multi-hop reasoning.

### ✨ Additional Key Features
- **Multi-Channel Retrieval**: Runs multiple strategies concurrently (Cortex activation, FAISS, BM25, KG entity walk, RAPTOR summaries), merging results through learned Reciprocal Rank Fusion.
- **Adaptive Routing & Self-Correction**: Intelligently skips retrieval for descriptive queries. Evaluates retrieved context and falls back to authoritative web searches if needed. 
- **Speculative Draft Generation (MoT)**: Partitions context and generates diverse drafts in parallel, synthesizing them for a comprehensive final response.
- **Built-In Security Layer**: Protects against prompt injections and immediately filters out PII data.

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

#### 1. Ingest Documents (Building the Cortex)

Before answering questions, ingest documents. The system will automatically build the initial chunk nodes, extract entities, and assign cortical columns to establish the baseline brain structure.
```bash
# Ingest an entire directory (supports .txt, .md, .html, .py, etc.)
gtr ingest /path/to/your/documents
```

#### 2. Query the CortexStore (Learning via Retrieval)

Query the knowledge base using the command-line interface. **With every query, the Hebbian learning mechanism updates synapse weights**. Use the `--verbose` flag to see the exact neural trace, cortical columns activated, and synapses strengthened.

```bash
# Complex comparison, printing the full neural trace
gtr query "Compare standard RAG to brain-inspired memory models" --verbose
```

#### 3. Start the API Server

Serve the application as a standalone REST container using FastAPI:
```bash
gtr serve --port 8000
```
Query it via `curl` or explore the Swagger UI at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does Hebbian plasticity improve retrieval?"}'
```

---

## Where Users Can Get Help

If you encounter issues, have questions, or need guidance on tuning hyperparameters for your specific data:

- **Issues / Bug Tracker**: Use the [GitHub Issues](https://github.com/saikiranreddy2710/graph-tree-rag/issues) page to report unexpected behavior.
- **Detailed Documentation**: For deep dives into the CortexStore pipeline, cortical columns configuration, and paper references, see our `docs/` folder. Check `ingestion/cortex_store.py` and `retrieval/cortex_retriever.py` for low-level mechanics.
- **Evaluation Benchmark**: Run the built-in benchmarking pipeline using `gtr evaluate` to track retrieval accuracy dynamically against standard RAG baselines.

---

## Who Maintains and Contributes

This project is authored and maintained by **[Sai Kiran Reddy](https://github.com/saikiranreddy2710)**. 

### Contribution Guidelines
We welcome community contributions! We are particularly interested in enhancements to learning rates, pruning algorithms, and scaling the CortexStore to efficiently handle **1M+ nodes**.
1. Fork the repo and create your feature branch.
2. Ensure you add or update relevant unit tests in the `tests/` directory.
3. Validate neural trace behaviors using the `--verbose` flag.
4. Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed coding standards.

### License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

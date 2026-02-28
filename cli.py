"""
CLI Interface — Typer-based command line for Graph-Tree Hybrid RAG.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from graph_tree_rag import __version__

app = typer.Typer(
    name="gtr",
    help="Graph-Tree Hybrid RAG v2.0 — Advanced Retrieval-Augmented Generation",
    add_completion=True,
)
console = Console()


def _run_async(coro):
    """Helper to run async functions from sync CLI."""
    return asyncio.get_event_loop().run_until_complete(coro)


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to file or directory to ingest"),
    glob: str = typer.Option("**/*.*", help="Glob pattern for directory ingestion"),
):
    """Ingest documents into the Graph-Tree RAG pipeline."""
    console.print(f"[bold blue]Ingesting from:[/] {path}")

    if not path.exists():
        console.print(f"[red]Error: Path not found: {path}[/]")
        raise typer.Exit(1)

    from orchestrator import GraphTreeRAG

    pipeline = GraphTreeRAG()

    with console.status("[bold green]Processing documents..."):
        stats = _run_async(pipeline.ingest(path, glob))

    table = Table(title="Ingestion Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


@app.command()
def query(
    question: str = typer.Argument(..., help="Your question"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full trace"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Query the Graph-Tree RAG pipeline."""
    from orchestrator import GraphTreeRAG

    pipeline = GraphTreeRAG()

    try:
        pipeline.load()
    except Exception as e:
        console.print(f"[red]Error loading indices: {e}[/]")
        console.print("[yellow]Run 'gtr ingest <path>' first to build indices[/]")
        raise typer.Exit(1)

    with console.status("[bold green]Thinking..."):
        response = _run_async(pipeline.query(question))

    if json_output:
        console.print_json(json.dumps(response.to_dict(), indent=2))
    else:
        # Rich display
        console.print(
            Panel(
                response.answer,
                title=f"[bold]Answer[/] (confidence: {response.confidence:.1%})",
                border_style="green",
            )
        )

        if response.sources:
            src_table = Table(title="Sources")
            src_table.add_column("#", width=3)
            src_table.add_column("Source", style="cyan")
            src_table.add_column("Score", width=8)
            src_table.add_column("Channel", width=10)
            src_table.add_column("Snippet", max_width=60)

            for i, src in enumerate(response.sources[:5], 1):
                src_table.add_row(
                    str(i),
                    src.source[:30],
                    f"{src.relevance_score:.3f}",
                    src.channel,
                    src.text_snippet[:60] + "...",
                )
            console.print(src_table)

        if verbose and response.trace:
            trace_table = Table(title="Retrieval Trace")
            trace_table.add_column("Step", style="cyan")
            trace_table.add_column("Detail", style="white")

            t = response.trace
            trace_table.add_row("Strategy", t.router_strategy)
            trace_table.add_row("Query Type", t.router_query_type)
            trace_table.add_row("Channels", ", ".join(t.channels_used))
            if t.crag_quality:
                trace_table.add_row("CRAG Quality", t.crag_quality)
            if t.speculative_strategies:
                trace_table.add_row("MoT Strategies", ", ".join(t.speculative_strategies))
            trace_table.add_row("Self-RAG Iterations", str(t.self_rag_iterations))
            trace_table.add_row("Latency", f"{t.total_latency_ms:.0f}ms")

            console.print(trace_table)

        if response.warnings:
            for w in response.warnings:
                console.print(f"[yellow]Warning: {w}[/]")


@app.command()
def evaluate(
    test_file: Path = typer.Option(None, help="Path to test queries JSON"),
    save_baseline: bool = typer.Option(False, help="Save results as new baseline"),
):
    """Run evaluation regression suite."""
    from evaluation.regression_suite import RegressionSuite

    suite = RegressionSuite(
        test_queries_path=test_file or Path("evaluation/test_queries/sample_queries.json"),
    )

    queries = suite.load_test_queries()
    console.print(f"[blue]Loaded {len(queries)} test queries[/]")
    console.print(
        "[yellow]Note: Full evaluation requires running the pipeline. "
        "Use the API or orchestrator for full eval.[/]"
    )


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold]Graph-Tree Hybrid RAG[/] v{__version__}")
    console.print(
        "Papers implemented: GraphRAG, RAPTOR, CRAG, Self-RAG, "
        "Speculative RAG, HyDE, Adaptive-RAG, Modular RAG"
    )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port number"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the FastAPI server."""
    import uvicorn

    console.print(f"[bold green]Starting server at http://{host}:{port}[/]")
    uvicorn.run("api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()

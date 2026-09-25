"""Command-line interface for the reference implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from evaldocai.demo import run_reference_demo
from evaldocai.extraction import extract_contract
from evaldocai.ingest import parse_pdf

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Evaluation-first OCR, extraction, retrieval, and benchmark demo.",
)
console = Console()


def _write_model(model: object, output: Path | None) -> None:
    payload = model.model_dump_json(indent=2)  # type: ignore[attr-defined]
    if output is None:
        console.print_json(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    console.print(f"Wrote [cyan]{output.resolve()}[/cyan]")


@app.command()
def parse(
    pdf: Annotated[Path, typer.Argument(exists=True, readable=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Parse a PDF into canonical page/block JSON."""

    _write_model(parse_pdf(pdf), output)


@app.command()
def extract(
    pdf: Annotated[Path, typer.Argument(exists=True, readable=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Extract the fixed contract schema with exact source evidence."""

    _write_model(extract_contract(parse_pdf(pdf)), output)


@app.command()
def demo(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for parsed JSON and reports."),
    ] = Path("artifacts/reference_run"),
) -> None:
    """Run the committed synthetic benchmark without an API key."""

    run = run_reference_demo(output_dir=output_dir)
    summary = run.report["summary"]
    table = Table(title="Eval-First Document AI — reference run")
    table.add_column("Quality gate")
    table.add_column("Result", justify="right", style="cyan")

    extraction = summary.get("extraction", {})
    layout = summary.get("layout", {})
    ocr = summary.get("ocr", {})
    retrieval = summary.get("retrieval", {})
    answer = summary.get("answer", {})
    table.add_row("Extraction exact match", f"{extraction.get('exact_match', 0):.1%}")
    table.add_row("Extraction token F1", f"{extraction.get('token_f1', 0):.1%}")
    table.add_row("OCR character error rate", f"{ocr.get('cer', 0):.1%}")
    table.add_row("Table cell accuracy", f"{layout.get('table_cell_accuracy', 0):.1%}")
    table.add_row("Two-column reading order", f"{layout.get('two_column_order_accuracy', 0):.1%}")
    table.add_row("Retrieval Recall@3", f"{retrieval.get('recall_at_k', {}).get('3', 0):.1%}")
    table.add_row("Retrieval MRR", f"{retrieval.get('mrr', 0):.1%}")
    table.add_row("Citation precision", f"{answer.get('citation_precision', 0):.1%}")
    table.add_row("Abstention accuracy", f"{answer.get('abstention_accuracy', 0):.1%}")
    console.print(table)
    console.print(f"\nJSON: [cyan]{run.output_dir / 'metrics.json'}[/cyan]")
    console.print(f"HTML: [cyan]{run.output_dir / 'report.html'}[/cyan]")


@app.command("show-metrics")
def show_metrics(
    report: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, dir_okay=False),
    ] = Path("artifacts/reference_run/metrics.json"),
) -> None:
    """Print a previously generated benchmark summary."""

    payload = json.loads(report.read_text(encoding="utf-8"))
    console.print_json(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    app()

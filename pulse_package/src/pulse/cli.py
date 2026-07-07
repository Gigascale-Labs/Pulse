'''Expose the package through the command line.'''

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pulse.config import PipelineConfig
from pulse.pipeline import run_pipeline


app = typer.Typer(
    no_args_is_help=True,
    help="Persona discovery pipeline.",
)


@app.command("run")
def run_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to a YAML pipeline configuration file.",
        ),
    ],
) -> None:
    """
    Run the persona discovery pipeline from a YAML config file.
    """

    pipeline_config = PipelineConfig.from_yaml(config)
    result = run_pipeline(pipeline_config)

    typer.echo("Pipeline completed.")
    typer.echo(f"Data hash: {result.metadata['data_hash']}")
    typer.echo(f"Prepared records: {len(result.prepared.posts)}")
    typer.echo(f"Embedding vectors: {result.embeddings.vectors.shape[0]}")
    typer.echo(f"Reduction spaces: {len(result.reductions)}")
    typer.echo(f"Cluster results: {len(result.clusters)}")
    typer.echo(f"Persona exports: {len(result.dumps)}")

    if result.dumps:
        typer.echo("Export directories:")
        for dump in result.dumps:
            typer.echo(f"  {dump.run_dir}")


@app.command("validate-config")
def validate_config_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to a YAML pipeline configuration file.",
        ),
    ],
) -> None:
    """
    Validate a YAML config file without running the pipeline.
    """

    PipelineConfig.from_yaml(config)
    typer.echo("Config is valid.")


def main() -> None:
    """
    Run the command-line interface.
    """

    app()
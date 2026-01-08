#!/usr/bin/env python3
"""
Wrapper script to process a dataset of protein structures by generating ESM embeddings and extracting binding site information.
"""

from pathlib import Path

import click
import torch

from extract_binding_info import extract_binding_info
from generate_esm_embeddings import generate_embeddings


@click.command()
@click.option(
    "--data-dir",
    "-p",
    required=True,
    type=click.Path(),
    default=Path.cwd() / "data" / "allosteric-sites" / "allosteric",
    help="Data folder containing a raw subfolder with the protein subfolders"
)
@click.option(
    "--jobs",
    "-j",
    default=1,
    type=int,
    help="Number of parallel jobs to pass to scripts"
)
@click.option(
    "--device",
    "-d",
    default="auto",
    type=click.Choice(["auto", "cpu", "cuda"]),
    help="Device to use for ESM embedding generation"
)
@click.option(
    "--batch",
    "-b",
    default=1,
    type=int,
    help="Batch size for ESM embedding generation"
)
@click.option(
    "--threshold",
    "-t",
    default=4.0,
    type=float,
    help="Distance threshold for binding site detection"
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force regeneration of embeddings and binding info even if they already exist"
)
@click.option(
    "--skip-depth",
    is_flag=True,
    help="Skip residue depth calculation (MSMS can hang on some structures)"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output (show individual errors in console)"
)
@click.pass_context
def main(
        ctx: click.Context,
        data_dir: Path,
        jobs: int,
        device: str,
        batch: int,
        threshold: float,
        force: bool,
        skip_depth: bool,
        verbose: bool,
) -> None:
    """
    Main function to process the dataset by generating ESM embeddings and extracting binding info.

    Parameters
    ----------
    ctx : click.Context
        Click context to invoke other commands.
    data_dir : Path
        Path to the data directory containing a 'raw' subdirectory.
    jobs : int
        Number of parallel jobs to use.
    device : str
        Device to use for ESM embedding generation ('auto', 'cpu', or 'cuda').
    batch : int
        Batch size for ESM embedding generation.
    threshold : float
        Distance threshold for binding site detection.
    force : bool
        Whether to force regeneration of embeddings and binding info.
    skip_depth : bool
        Whether to skip residue depth calculation.
    verbose : bool
        Whether to show individual errors in console.

    Returns
    -------
    None
    """
    data_dir = Path(data_dir).resolve()
    data_root = data_dir / "raw"
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    if batch <= 0:
        raise SystemExit(f"Batch size must be a positive integer, got: {batch}")

    resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {resolved_device}")

    print("\n" + "=" * 60)
    print(f"Processing dataset: {data_root}")
    print("=" * 60)

    # Generate ESM embeddings using Click's ctx.invoke()
    print("\nGenerating ESM embeddings...")
    try:
        ctx.invoke(
            generate_embeddings,
            path=data_root,
            model="esm2_t33_650M_UR50D",
            output_format="npz",
            batch_size=batch,
            n_jobs=jobs if resolved_device != "cuda" else 1,
            verbose=verbose,
            monitor_memory=True,
            device=resolved_device,
            force=force,
        )
    except SystemExit as e:
        if e.code != 0:
            print(f"Warning: Some embeddings failed (exit code {e.code})")

    # Extract binding info using Click's ctx.invoke()
    print("\nExtracting binding info...")
    try:
        ctx.invoke(
            extract_binding_info,
            path=data_root,
            n_jobs=jobs,
            threshold=threshold,
            verbose=verbose,
            backend="processes",
            force=force,
            skip_depth=skip_depth,
        )
    except SystemExit as e:
        if e.code != 0:
            print(f"Warning: Some binding info extractions failed (exit code {e.code})")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Portable Python wrapper for scripts/process_data.sh
Runs generate_esm_embeddings.py and extract_binding_info.py for every folder under data/data
"""
from pathlib import Path
import subprocess
import sys
import torch
import click


def run_command(cmd, cwd=None):
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


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
def main(
        data_dir: Path,
        jobs: int,
        device: Path,
        batch: int,
        threshold: float
):
    data_root = data_dir / "raw"
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    script_root = Path(__file__).parent.resolve()
    gen_script = script_root / "generate_esm_embeddings.py"
    extract_binding_info_script = script_root / "extract_binding_info.py"

    for script in [gen_script, extract_binding_info_script]:
        if not script.exists():
            raise SystemExit(f"Script not found: {script}")

    if batch <= 0:
        raise SystemExit(f"Batch size must be a positive integer, got: {batch}")

    resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {resolved_device}")

    print("\n" + "=" * 60)
    print(f"Processing dataset: {data_root}")
    print("=" * 60)

    print("Generating ESM embeddings...")
    esm_jobs = jobs if resolved_device != "cuda" else 1
    run_command([
        sys.executable,
        str(gen_script),
        "-p", str(data_root),
        "-j", str(esm_jobs),
        "-d", resolved_device,
        "-b", str(batch)
    ])

    print("Extracting binding info...")
    run_command([
        sys.executable,
        str(extract_binding_info_script),
        "-p", str(data_root),
        "-j", str(jobs),
        "-t", str(threshold),
        "-b", "processes"
    ])


if __name__ == '__main__':
    main()

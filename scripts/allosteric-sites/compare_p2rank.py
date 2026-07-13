#!/usr/bin/env python3
"""
Run P2Rank on an ASD split and report DCC/DCA success rates comparable to VN-EGNN.

For each protein in a split id file, this runs P2Rank pocket prediction, ranks the
predicted pocket centers by P2Rank score, and computes Distance-to-Closest-Center
(DCC) and Distance-to-Closest-Atom (DCA) at a 4.0 Angstrom threshold across rank-n
thresholds -- the exact same success definition used in
src/utils/misc.evaluate_protein_predictions, so the numbers line up with the VN-EGNN
evaluation table.

Requires P2Rank (`prank`) on PATH: https://github.com/rdk/p2rank

Example:
  python scripts/allosteric-sites/compare_p2rank.py \
      --asd-dir data/allosteric-sites/allosteric \
      --split-file data/allosteric-sites/allosteric/splits/test_ids_allosteric_mmseqs30 \
      --threshold 4.0 --num-ranks 8
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm


def run_p2rank(protein_paths: dict[str, Path], outdir: Path, threads: int) -> Path:
    """Stage uniquely-named symlinks (all inputs are named protein.pdb otherwise) and
    run a single batched `prank predict`."""
    stage = outdir / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    ds_lines = []
    for pdb_id, path in protein_paths.items():
        link = stage / f"{pdb_id}.pdb"
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            link.symlink_to(path.resolve())
        except OSError:
            shutil.copy(path, link)
        ds_lines.append(link.name)

    ds_file = stage / "proteins.ds"
    ds_file.write_text("\n".join(ds_lines) + "\n")

    pred_out = outdir / "predictions"
    cmd = ["prank", "predict", str(ds_file), "-o", str(pred_out), "-threads", str(threads)]
    click.echo(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return pred_out


def load_p2rank_centers(pred_csv: Path) -> np.ndarray:
    """Return pocket centers (N, 3) ordered by P2Rank rank (best first)."""
    df = pd.read_csv(pred_csv, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    if "rank" in df.columns:
        df = df.sort_values("rank")
    return df[["center_x", "center_y", "center_z"]].to_numpy(dtype=float)


def evaluate_protein(pred_coords: np.ndarray, binding: np.lib.npyio.NpzFile,
                     num_ranks: int, threshold: float) -> dict:
    """DCC/DCA hits per rank-n for one protein, mirroring evaluate_protein_predictions."""
    centers = binding["binding_site_centers"]
    lig_ids = np.unique(binding["ligand_ids"])
    num_ligs = len(lig_ids)

    dca = {f"n_rank_dca_{i}": 0 for i in range(num_ranks)}
    dcc = {f"n_rank_dcc_{i}": 0 for i in range(num_ranks)}

    if len(pred_coords) == 0:
        return {"num_ligs": num_ligs, **dca, **dcc}

    for lig_id in lig_ids:
        lig_coords = binding["ligand_coords"][binding["ligand_ids"] == lig_id]
        for i in range(num_ranks):
            top = pred_coords[: i + num_ligs]
            dist = np.linalg.norm(lig_coords[:, None] - top, axis=-1)
            dca[f"n_rank_dca_{i}"] += int((dist <= threshold).any())

    for center in centers:
        for i in range(num_ranks):
            top = pred_coords[: i + num_ligs]
            dist = np.linalg.norm(center - top, axis=-1)
            dcc[f"n_rank_dcc_{i}"] += int((dist <= threshold).any())

    return {"num_ligs": num_ligs, **dca, **dcc}


@click.command()
@click.option("--asd-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--split-file", required=True, type=click.Path(exists=True, path_type=Path),
              help="File with one PDB id per line (e.g. test_ids_allosteric_mmseqs30).")
@click.option("--threshold", default=4.0, type=float)
@click.option("--num-ranks", default=8, type=int, help="rank-0 .. rank-(N-1).")
@click.option("--threads", default=4, type=int)
@click.option("--outdir", default=None, type=click.Path(path_type=Path),
              help="Where to put P2Rank output (default: a temp dir).")
def main(asd_dir, split_file, threshold, num_ranks, threads, outdir):
    asd_dir = Path(asd_dir)
    raw = asd_dir / "raw"
    ids = [x.strip() for x in Path(split_file).read_text().split() if x.strip()]

    protein_paths = {}
    for pdb_id in ids:
        p = raw / pdb_id / "protein.pdb"
        if p.exists():
            protein_paths[pdb_id] = p
    click.echo(f"[INFO] {len(protein_paths)}/{len(ids)} proteins have protein.pdb")

    outdir = Path(outdir) if outdir else Path(tempfile.mkdtemp(prefix="p2rank_"))
    pred_out = run_p2rank(protein_paths, outdir, threads)

    rows = []
    for pdb_id in tqdm(protein_paths, desc="scoring"):
        pred_csv = pred_out / f"{pdb_id}.pdb_predictions.csv"
        binding_path = raw / pdb_id / "binding.npz"
        if not pred_csv.exists() or not binding_path.exists():
            continue
        try:
            pred_coords = load_p2rank_centers(pred_csv)
            with np.load(binding_path) as binding:
                res = evaluate_protein(pred_coords, binding, num_ranks, threshold)
            rows.append({"protein_name": pdb_id, **res})
        except Exception as e:  # noqa: BLE001
            click.echo(f"[WARN] {pdb_id}: {e}")

    df = pd.DataFrame(rows)
    click.echo(f"[INFO] Scored {len(df)} proteins")

    # Per-protein success ratio (hits / num_ligs), then mean across proteins -- same
    # aggregation as src/utils/misc.compute_metric_ratios.
    print("\nP2Rank on", split_file.name)
    print(f"{'rank':>5} {'DCC':>8} {'DCA':>8}")
    for i in range(num_ranks):
        dcc = (df[f"n_rank_dcc_{i}"] / df["num_ligs"]).clip(upper=1.0).mean()
        dca = (df[f"n_rank_dca_{i}"] / df["num_ligs"]).clip(upper=1.0).mean()
        print(f"{i:>5} {dcc:>8.3f} {dca:>8.3f}")

    csv_out = outdir / "p2rank_metrics.csv"
    df.to_csv(csv_out, index=False)
    click.echo(f"[INFO] Per-protein metrics -> {csv_out}")


if __name__ == "__main__":
    main()

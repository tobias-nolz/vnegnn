#!/usr/bin/env python3
"""
Run P2Rank on an ASD split and report DCC/DCA success rates comparable to VN-EGNN.

For each protein in a split id file, this runs P2Rank pocket prediction, ranks the
predicted pocket centers by P2Rank score, and computes Distance-to-Closest-Center
(DCC) and Distance-to-Closest-Atom (DCA) at a 4.0 Angstrom threshold across rank-n
thresholds -- the exact same success definition used in
src/utils/misc.evaluate_protein_predictions, so the numbers line up with the VN-EGNN
evaluation table.

Requires P2Rank (`prank`) on PATH (or pass --prank-bin): https://github.com/rdk/p2rank

Two modes:
  * default (full-assembly, unfiltered): scores against every site in binding.npz, on the
    raw full-assembly protein.pdb. Comparable to the *unfiltered* zero-shot figure only.
  * --matched: reproduces the denominator of the joint-model rows in the thesis
    (tab:joint-results). It (a) restricts to the exact protein set the VN-EGNN matched eval
    scored (via --proteins-csv, the model's predictions_allosteric.csv), (b) runs P2Rank on
    the single-chain host structure each VN-EGNN single-chain run saw, and (c) scores against
    the identical ground truth: 8 A max_center_dist (full res_coords) + allosteric-only
    (site_type==1), aggregated per-protein with the same compute_metric_ratios. This is the
    number that belongs in tab:joint-results.

Examples:
  # full-assembly, unfiltered (original behaviour)
  python scripts/allosteric-sites/compare_p2rank.py \
      --asd-dir data/allosteric-sites/allosteric \
      --split-file data/allosteric-sites/allosteric/splits/test_ids_allosteric_mmseqs30

  # matched to tab:joint-results (single-chain input, allosteric 8 A denominator)
  python scripts/allosteric-sites/compare_p2rank.py \
      --asd-dir data/allosteric-sites/allosteric \
      --split-file data/allosteric-sites/allosteric/splits/test_ids_allosteric_mmseqs30 \
      --matched \
      --proteins-csv logs/eval/runs/2026-07-28_14-08-10-320556/predictions_allosteric.csv \
      --prank-bin /home/user/opt/p2rank_2.5.1/prank
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Reuse the codebase's own single-chain selection and metric aggregation so the matched
# comparison is byte-identical to the VN-EGNN eval, not a re-implementation.
from src.datasets.binding_dataset import select_single_chain  # noqa: E402
from src.utils.misc import compute_metric_ratios  # noqa: E402


def run_p2rank(protein_paths: dict[str, Path], outdir: Path, threads: int,
               prank_bin: str, single_chain: bool) -> Path:
    """Stage uniquely-named PDBs (all inputs are named protein.pdb otherwise) and run a
    single batched `prank predict`. When single_chain, each staged PDB is already reduced
    to its host chain (see stage_single_chain)."""
    stage = outdir / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    ds_lines = []
    for pdb_id, path in protein_paths.items():
        link = stage / f"{pdb_id}.pdb"
        if link.exists() or link.is_symlink():
            link.unlink()
        if single_chain:
            # `path` already points at a materialised single-chain PDB; copy it in.
            shutil.copy(path, link)
        else:
            try:
                link.symlink_to(path.resolve())
            except OSError:
                shutil.copy(path, link)
        ds_lines.append(link.name)

    ds_file = stage / "proteins.ds"
    ds_file.write_text("\n".join(ds_lines) + "\n")

    pred_out = outdir / "predictions"
    cmd = [prank_bin, "predict", str(ds_file), "-o", str(pred_out), "-threads", str(threads)]
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


def _host_chain(pid: str, binding) -> str:
    """The single host chain VN-EGNN's single-chain extraction keeps for this protein --
    derived from select_single_chain so the P2Rank input matches the model input exactly."""
    residue_keep, _ = select_single_chain(
        protein_name=pid,
        coords=binding["res_coords"],
        chains=binding["chains"],
        binding_sites=binding["binding_site_centers"],
        ligand_coords=binding["ligand_coords"],
        ligand_ids=binding["ligand_ids"],
        site_types=binding["site_types"] if "site_types" in binding.files else None,
    )
    kept = np.unique(binding["chains"][residue_keep])
    return str(kept[0])


def write_single_chain_pdb(src_pdb: Path, anchor: str, dst: Path) -> None:
    """Write a copy of src_pdb keeping only ATOM/HETATM/TER records on chain `anchor`;
    non-coordinate records (HEADER, CRYST1, ...) are preserved so P2Rank sees a valid file."""
    out = []
    for ln in src_pdb.read_text().splitlines():
        rec = ln[:6].strip()
        if rec in ("ATOM", "HETATM", "TER", "ANISOU"):
            if len(ln) > 21 and ln[21] == anchor:
                out.append(ln)
        else:
            out.append(ln)
    dst.write_text("\n".join(out) + "\n")


def _matched_ground_truth(binding, max_center_dist: float, site_type_filter: int):
    """Reproduce the ground-truth filtering of src/utils/misc.evaluate_protein_predictions
    exactly: drop centers > max_center_dist from any (full-assembly) residue, then keep only
    sites of the requested type. Returns (centers, ligand_coords, ligand_ids) or None when no
    site of the requested type survives (so the protein is skipped, matching the model eval)."""
    centers = binding["binding_site_centers"]
    ligand_coords = binding["ligand_coords"]
    ligand_ids = binding["ligand_ids"]
    site_types = binding["site_types"] if "site_types" in binding.files else None
    res_coords = binding["res_coords"]

    if max_center_dist is not None and len(centers):
        reachable = (
            np.linalg.norm(centers[:, None, :] - res_coords[None, :, :], axis=-1).min(axis=1)
            <= max_center_dist
        )
        if not reachable.all():
            remap = np.full(len(reachable), -1, dtype=np.int64)
            remap[reachable] = np.arange(int(reachable.sum()))
            atom_keep = reachable[ligand_ids]
            ligand_coords = ligand_coords[atom_keep]
            ligand_ids = remap[ligand_ids[atom_keep]]
            centers = centers[reachable]
            if site_types is not None:
                site_types = site_types[reachable]
            if len(centers) == 0:
                return None

    if site_type_filter is not None and site_types is not None:
        center_mask = site_types == site_type_filter
        atom_mask = site_types[ligand_ids] == site_type_filter
        centers = centers[center_mask]
        ligand_coords = ligand_coords[atom_mask]
        ligand_ids = ligand_ids[atom_mask]
        if len(ligand_ids) == 0 and len(centers) == 0:
            return None

    return centers, ligand_coords, ligand_ids


def evaluate_protein(pred_coords: np.ndarray, centers: np.ndarray, ligand_coords: np.ndarray,
                     ligand_ids: np.ndarray, num_ranks: int, threshold: float) -> dict:
    """DCC/DCA hits per rank-n for one protein, mirroring evaluate_protein_predictions: the
    rank-n pool is the top (num_sites + n) P2Rank pockets (already score-ranked)."""
    lig_ids = np.unique(ligand_ids)
    num_ligs = len(lig_ids)

    dca = {f"n_rank_dca_{i}": 0 for i in range(num_ranks)}
    dcc = {f"n_rank_dcc_{i}": 0 for i in range(num_ranks)}

    if len(pred_coords) == 0:
        return {"num_ligs": num_ligs, **dca, **dcc}

    for lig_id in lig_ids:
        lig = ligand_coords[ligand_ids == lig_id]
        for i in range(num_ranks):
            top = pred_coords[: i + num_ligs]
            dist = np.linalg.norm(lig[:, None] - top, axis=-1)
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
@click.option("--prank-bin", default="prank", type=str, help="P2Rank launcher (default: prank on PATH).")
@click.option("--matched", is_flag=True, default=False,
              help="Reproduce the tab:joint-results denominator (single-chain input, allosteric "
                   "8 A filter, per-protein aggregation). See module docstring.")
@click.option("--proteins-csv", default=None, type=click.Path(path_type=Path),
              help="[--matched] The VN-EGNN matched eval's predictions_allosteric.csv; its "
                   "protein_name column defines the exact protein set to score.")
@click.option("--max-center-dist", default=8.0, type=float, help="[--matched] ground-truth site filter.")
@click.option("--outdir", default=None, type=click.Path(path_type=Path),
              help="Where to put P2Rank output (default: a temp dir).")
def main(asd_dir, split_file, threshold, num_ranks, threads, prank_bin, matched,
         proteins_csv, max_center_dist, outdir):
    asd_dir = Path(asd_dir)
    raw = asd_dir / "raw"
    ids = [x.strip() for x in Path(split_file).read_text().split() if x.strip()]

    if matched and proteins_csv is not None:
        wanted = set(pd.read_csv(proteins_csv)["protein_name"].unique())
        ids = [i for i in ids if i in wanted]
        click.echo(f"[INFO] --matched: restricted to {len(ids)} proteins from {Path(proteins_csv).name}")

    outdir = Path(outdir) if outdir else Path(tempfile.mkdtemp(prefix="p2rank_"))
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve input structures. In matched mode, materialise single-chain PDBs (host chain
    # from select_single_chain) so P2Rank sees the same structure the model did.
    protein_paths: dict[str, Path] = {}
    if matched:
        sc_dir = outdir / "single_chain"
        sc_dir.mkdir(parents=True, exist_ok=True)
        for pid in ids:
            p = raw / pid / "protein.pdb"
            b_path = raw / pid / "binding.npz"
            if not (p.exists() and b_path.exists()):
                continue
            with np.load(b_path) as b:
                try:
                    anchor = _host_chain(pid, b)
                except Exception as e:  # noqa: BLE001
                    click.echo(f"[WARN] {pid}: host-chain selection failed: {e}")
                    continue
            dst = sc_dir / f"{pid}.pdb"
            write_single_chain_pdb(p, anchor, dst)
            protein_paths[pid] = dst
    else:
        for pid in ids:
            p = raw / pid / "protein.pdb"
            if p.exists():
                protein_paths[pid] = p
    click.echo(f"[INFO] {len(protein_paths)}/{len(ids)} proteins have a usable structure")

    pred_out = run_p2rank(protein_paths, outdir, threads, prank_bin, single_chain=matched)

    rows = []
    for pdb_id in tqdm(protein_paths, desc="scoring"):
        pred_csv = pred_out / f"{pdb_id}.pdb_predictions.csv"
        binding_path = raw / pdb_id / "binding.npz"
        if not pred_csv.exists() or not binding_path.exists():
            continue
        try:
            pred_coords = load_p2rank_centers(pred_csv)
            with np.load(binding_path) as binding:
                if matched:
                    gt = _matched_ground_truth(binding, max_center_dist, site_type_filter=1)
                    if gt is None:
                        continue  # no allosteric site survives the filter -> skip (as model eval does)
                    centers, ligand_coords, ligand_ids = gt
                else:
                    centers = binding["binding_site_centers"]
                    ligand_coords = binding["ligand_coords"]
                    ligand_ids = binding["ligand_ids"]
                res = evaluate_protein(pred_coords, centers, ligand_coords, ligand_ids,
                                       num_ranks, threshold)
            rows.append({"protein_name": pdb_id, **res})
        except Exception as e:  # noqa: BLE001
            click.echo(f"[WARN] {pdb_id}: {e}")

    df = pd.DataFrame(rows)
    click.echo(f"[INFO] Scored {len(df)} proteins")

    mode = "matched (single-chain, allosteric 8 A)" if matched else "full-assembly, unfiltered"
    print(f"\nP2Rank on {split_file.name} -- {mode}")
    print(f"{'rank':>5} {'DCC':>8} {'DCA':>8}")
    if matched:
        # Byte-identical aggregation to the VN-EGNN eval: per-protein round(hits/num_ligs, 2)
        # via compute_metric_ratios, then mean over proteins.
        _, dcc_r, dcc_cols = compute_metric_ratios(df.copy(), "dcc")
        _, dca_r, dca_cols = compute_metric_ratios(df.copy(), "dca")
        dcc_mean = dcc_r[dcc_cols].mean(axis=0)
        dca_mean = dca_r[dca_cols].mean(axis=0)
        for i in range(num_ranks):
            print(f"{i:>5} {dcc_mean[f'dcc_ratio_n_rank_dcc_{i}']:>8.3f} "
                  f"{dca_mean[f'dca_ratio_n_rank_dca_{i}']:>8.3f}")
    else:
        for i in range(num_ranks):
            dcc = (df[f"n_rank_dcc_{i}"] / df["num_ligs"]).clip(upper=1.0).mean()
            dca = (df[f"n_rank_dca_{i}"] / df["num_ligs"]).clip(upper=1.0).mean()
            print(f"{i:>5} {dcc:>8.3f} {dca:>8.3f}")

    csv_out = outdir / "p2rank_metrics.csv"
    df.to_csv(csv_out, index=False)
    click.echo(f"[INFO] Per-protein metrics -> {csv_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create sequence-identity-based train/val/test splits for the ASD allosteric dataset.

Clusters all ASD protein sequences (plus, optionally, the sc-PDB train/valid
sequences) with mmseqs2 at a given identity threshold, then assigns *whole clusters*
to train/valid/test. This guarantees that:

  * no ASD test (or valid) protein is homologous to any ASD train protein, and
  * no ASD test/valid protein is homologous to any sc-PDB training protein
    (cross-dataset leakage into the jointly-trained orthosteric side).

ASD proteins that share a cluster with a sc-PDB train/valid protein are therefore
kept out of the ASD test/valid folds and routed to the ASD train fold instead
(homology to orthosteric *training* data is harmless for training).

Output (written under <asd-dir>/splits/):
  train_ids_allosteric_<suffix>
  valid_ids_allosteric_<suffix>
  test_ids_allosteric_<suffix>

Requires `mmseqs` on PATH (https://github.com/soedinglab/MMseqs2), unless you pass a
precomputed --cluster-tsv.

Example:
  python scripts/allosteric-sites/make_splits.py \
      --asd-dir data/allosteric-sites/allosteric \
      --scpdb-dir data/data/sc-pdb \
      --min-seq-id 0.3 --coverage 0.8 --suffix mmseqs30
"""
import random
import subprocess
import sys
from pathlib import Path

import click
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

# `ASD:` / `SCPDB:` prefixes tag each fasta record with its source so the cluster
# assignment can tell allosteric from orthosteric members. The original id follows.
ASD_TAG = "ASD:"
SCPDB_TAG = "SCPDB:"


def longest_chain_sequence(pdb_path: Path) -> str:
    """Return the single longest standard-residue chain sequence of a PDB file.

    Uses the same residue filtering as scripts/extract_binding_info.py (standard
    residues only, seq1-convertible) so sequences are consistent with the model input.
    """
    structure = PDBParser(QUIET=True).get_structure("p", str(pdb_path))
    best = ""
    for model in structure:
        for chain in model:
            residues = []
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                try:
                    residues.append(seq1(residue.resname))
                except KeyError:
                    continue
            seq = "".join(residues)
            if len(seq) > len(best):
                best = seq
        break  # first model only
    return best


def _seq_record(folder: Path, tag: str) -> tuple[str, str] | None:
    protein = folder / "protein.pdb"
    if not protein.exists():
        return None
    try:
        seq = longest_chain_sequence(protein)
    except Exception:
        return None
    if not seq:
        return None
    return (f"{tag}{folder.name}", seq)


def gather_sequences(raw_dir: Path, tag: str, only_ids: set[str] | None, n_jobs: int):
    folders = [d for d in raw_dir.iterdir() if d.is_dir()]
    if only_ids is not None:
        folders = [d for d in folders if d.name in only_ids]
    records = Parallel(n_jobs=n_jobs)(
        delayed(_seq_record)(d, tag)
        for d in tqdm(folders, desc=f"seqs[{tag.rstrip(':')}]")
    )
    return [r for r in records if r is not None]


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with open(path, "w") as f:
        for name, seq in records:
            f.write(f">{name}\n{seq}\n")


def run_mmseqs(fasta: Path, out_prefix: Path, min_seq_id: float, coverage: float) -> Path:
    tmp = out_prefix.parent / "mmseqs_tmp"
    cmd = [
        "mmseqs", "easy-cluster", str(fasta), str(out_prefix), str(tmp),
        "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0",
    ]
    click.echo(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return Path(f"{out_prefix}_cluster.tsv")


def parse_clusters(cluster_tsv: Path) -> dict[str, list[str]]:
    """mmseqs *_cluster.tsv is: <representative>\\t<member> per line."""
    clusters: dict[str, list[str]] = {}
    with open(cluster_tsv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rep, member = line.split("\t")[:2]
            clusters.setdefault(rep, []).append(member)
    return clusters


def assign_splits(clusters: dict[str, list[str]], ratios: tuple[float, float, float], seed: int):
    """Assign whole clusters to train/valid/test.

    Clusters containing any sc-PDB member are forced to train (their ASD members are
    homologous to orthosteric training data and must not leak into valid/test).
    """
    train, valid, test = [], [], []
    forced_train_clusters = []
    free_clusters = []

    for rep, members in clusters.items():
        asd_ids = [m[len(ASD_TAG):] for m in members if m.startswith(ASD_TAG)]
        if not asd_ids:
            continue  # sc-PDB-only cluster, nothing to assign
        has_scpdb = any(m.startswith(SCPDB_TAG) for m in members)
        if has_scpdb:
            forced_train_clusters.append(asd_ids)
        else:
            free_clusters.append(asd_ids)

    rng = random.Random(seed)
    rng.shuffle(free_clusters)

    r_train, r_valid, _ = ratios
    n = len(free_clusters)
    n_train = int(n * r_train)
    n_valid = int(n * r_valid)

    for i, asd_ids in enumerate(free_clusters):
        if i < n_train:
            train.extend(asd_ids)
        elif i < n_train + n_valid:
            valid.extend(asd_ids)
        else:
            test.extend(asd_ids)
    for asd_ids in forced_train_clusters:
        train.extend(asd_ids)

    return sorted(set(train)), sorted(set(valid)), sorted(set(test)), len(forced_train_clusters)


@click.command()
@click.option("--asd-dir", required=True, type=click.Path(exists=True, path_type=Path),
              help="ASD dataset dir containing raw/ and splits/.")
@click.option("--scpdb-dir", default=None, type=click.Path(path_type=Path),
              help="sc-PDB dataset dir (with raw/ and splits/) to remove cross-leakage.")
@click.option("--min-seq-id", default=0.3, type=float, help="mmseqs2 --min-seq-id (default 0.3).")
@click.option("--coverage", default=0.8, type=float, help="mmseqs2 -c coverage (default 0.8).")
@click.option("--ratios", default=(0.8, 0.1, 0.1), type=(float, float, float),
              help="train/valid/test cluster ratios (default 0.8 0.1 0.1).")
@click.option("--suffix", default="mmseqs30", help="Split id filename suffix (default mmseqs30).")
@click.option("--seed", default=42, type=int)
@click.option("--jobs", "-j", default=cpu_count() - 1, type=int)
@click.option("--cluster-tsv", default=None, type=click.Path(path_type=Path),
              help="Use a precomputed mmseqs *_cluster.tsv instead of running mmseqs.")
def main(asd_dir, scpdb_dir, min_seq_id, coverage, ratios, suffix, seed, jobs, cluster_tsv):
    asd_dir = Path(asd_dir)
    splits_dir = asd_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    work = splits_dir / "_mmseqs"
    work.mkdir(exist_ok=True)

    if cluster_tsv is None:
        records = gather_sequences(asd_dir / "raw", ASD_TAG, only_ids=None, n_jobs=jobs)
        click.echo(f"[INFO] Collected {len(records)} ASD sequences")

        if scpdb_dir is not None:
            scpdb_dir = Path(scpdb_dir)
            scpdb_ids: set[str] = set()
            for m in ("train", "valid"):
                ids_file = scpdb_dir / "splits" / f"{m}_ids_scpdb"
                if ids_file.exists():
                    scpdb_ids.update(ids_file.read_text().split())
            scpdb_records = gather_sequences(
                scpdb_dir / "raw", SCPDB_TAG, only_ids=scpdb_ids, n_jobs=jobs
            )
            click.echo(f"[INFO] Collected {len(scpdb_records)} sc-PDB train/valid sequences")
            records += scpdb_records

        fasta = work / "combined.fasta"
        write_fasta(records, fasta)
        cluster_tsv = run_mmseqs(fasta, work / "cluster", min_seq_id, coverage)

    clusters = parse_clusters(Path(cluster_tsv))
    train, valid, test, n_forced = assign_splits(clusters, ratios, seed)

    for name, ids in (("train", train), ("valid", valid), ("test", test)):
        out = splits_dir / f"{name}_ids_allosteric_{suffix}"
        out.write_text("\n".join(ids) + "\n")
        click.echo(f"[INFO] Wrote {len(ids)} ids -> {out}")
    click.echo(f"[INFO] {n_forced} ASD clusters routed to train due to sc-PDB homology")

    # Sanity: splits must be disjoint.
    assert not (set(train) & set(valid)), "train/valid overlap"
    assert not (set(train) & set(test)), "train/test overlap"
    assert not (set(valid) & set(test)), "valid/test overlap"
    click.echo("[INFO] Splits are disjoint. Done.")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Setup data for VN-EGNN allosteric site prediction.
This includes downloading PDB files and extracting ligand information.
"""
import pandas as pd
from pathlib import Path

from prepare_pdb import prepare_pdb_directory
from prepare_ligand import prepare_ligands_from_asd
from tqdm import tqdm
import click


def setup_data(
        output_dir: Path,
        asd_dataset: pd.DataFrame,
        n_jobs: int,
        skip_existing: bool = False,
) -> None:
    """
    Setup data for VN-EGNN allosteric site prediction.

    This will:
      - download PDB files into `pdb_dir` (one folder per PDB with `protein.pdb` inside)
      - extract ligand PDBs from the downloaded protein files

    :param output_dir: Path to the directory where PDB files will be stored.
    :param asd_dataset: DataFrame containing ASD dataset information.
    :param n_jobs: Number of parallel workers for downloads/extraction.
    :param skip_existing: If True, skip creating files that already exist.
    :return: None
    """
    output_dir = Path(output_dir)

    if asd_dataset is None or asd_dataset.empty:
        raise ValueError("ASD dataset is empty or not provided")

    # Extract PDB ids from the ASD dataset
    if 'allosteric_pdb' not in asd_dataset.columns:
        raise KeyError("ASD dataset must contain column 'allosteric_pdb'")

    pdb_ids = asd_dataset['allosteric_pdb'].tolist()
    tqdm.write(f"Preparing PDB directory at {output_dir} with {len(pdb_ids)} PDBs (workers={n_jobs})")
    prepare_pdb_directory(
        pdb_dir=output_dir,
        pdb_ids=pdb_ids,
        n_jobs=n_jobs
    )

    if not {'modulator_chain', 'modulator_resi'}.issubset(asd_dataset.columns):
        raise KeyError("ASD dataset must contain columns 'modulator_chain' and 'modulator_resi' for ligand extraction")

    ligand_info = pd.DataFrame(
        {
            'pdb_id': asd_dataset['allosteric_pdb'],
            'ligand_chain': asd_dataset['modulator_chain'],
            'ligand_residue': asd_dataset['modulator_resi'],
        }
    )
    tqdm.write(f"Extracting ligands (skip_existing={skip_existing}, workers={n_jobs})")
    prepare_ligands_from_asd(
        pdb_dir=output_dir,
        ligand_info=ligand_info,
        skip_existing=skip_existing,
        workers=n_jobs
    )


def _read_asd_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ASD dataset file not found: {path}")
    return pd.read_csv(path, sep=None, engine='python')


def setup_splits(
        output_dir: Path,
        raw_dir: Path,
):
    """
    Setup up data splits as all test data.
    TODO: Setup dataset splits given training protein data.
    :param output_dir: Directory where split files will be stored
    :param raw_dir: Directory containing raw PDB data
    :return: None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pdbs = [pdb_dir.name for pdb_dir in raw_dir.iterdir() if pdb_dir.is_dir()]

    with open(output_dir / "test_ids_allosteric", "w") as f:
        for pdb_id in all_pdbs:
            f.write(f"{pdb_id}\n")


@click.command()
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(),
    help="Directory where PDB folders will be stored (one folder per PDB)",
    default=Path.cwd() / "data" / "allosteric-sites" / "allosteric"
)
@click.option(
    "--asd-file",
    "-a",
    required=True,
    type=click.Path(),
    help=("Path to ASD dataset file (CSV/TSV) containing columns 'allosteric_pdb','modulator_chain','modulator_resi'")
)
@click.option(
    "--jobs",
    "-j",
    default=1,
    type=int,
    help="Number of parallel workers"
)
@click.option(
    "--no-skip",
    is_flag=True,
    default=False,
    help="Do not skip existing extracted ligand files"
)
def main(
        output_dir: Path,
        asd_file: Path,
        jobs: int,
        no_skip: bool
):
    print(f"Loading ASD dataset from: {asd_file}")
    asd_df = _read_asd_dataset(asd_file)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / 'raw'
    setup_data(
        output_dir=raw_dir,
        asd_dataset=asd_df,
        n_jobs=jobs,
        skip_existing=not no_skip
    )

    split_dir = output_dir / 'splits'
    setup_splits(
        output_dir=split_dir,
        raw_dir=raw_dir
    )


if __name__ == '__main__':
    main()

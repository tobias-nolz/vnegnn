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
        force_ligand_extraction: bool = False,
        clear_existing_pdb: bool = False,
        max_diff: int = 0,
        verbose: bool = False
) -> None:
    """
    Setup data for VN-EGNN allosteric site prediction.

    This will:
      - download PDB files into `pdb_dir` (one folder per PDB with `protein.pdb` inside)
      - extract ligand PDBs from the downloaded protein files

    Parameters
    ----------
    output_dir : Path
        Directory where PDB folders will be stored (one folder per PDB)
    asd_dataset : pd.DataFrame
        ASD dataset containing columns 'allosteric_pdb', 'modulator_chain', 'modulator_resi'
    n_jobs : int
        Number of parallel workers
    force_ligand_extraction : bool
        If True, force re-extraction of ligand files even if they already exist
    clear_existing_pdb : bool
        If True, clear existing PDB files before downloading new ones
    max_diff : int
        Maximum residue ID difference for fuzzy matching (0 = exact match only, 2 = allow ±2)
    verbose : bool
        If True, enable verbose output

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the ASD dataset is empty or no valid entries remain after filtering missing values
    KeyError
        If required columns are missing from the ASD dataset
    """
    output_dir = Path(output_dir)

    if asd_dataset is None or asd_dataset.empty:
        raise ValueError("ASD dataset is empty or not provided")

    # Validate required columns
    required_columns = {'allosteric_pdb', 'modulator_chain', 'modulator_resi'}
    missing_columns = required_columns - set(asd_dataset.columns)
    if missing_columns:
        raise KeyError(f"ASD dataset is missing required columns: {missing_columns}")

    # Filter out rows with missing PDB IDs or ligand info
    initial_count = len(asd_dataset)
    mask = asd_dataset[['allosteric_pdb', 'modulator_chain', 'modulator_resi']].isna().any(axis=1)
    filtered_rows = asd_dataset[mask]
    asd_dataset = asd_dataset[~mask]
    filtered_count = initial_count - len(asd_dataset)

    if filtered_count > 0:
        tqdm.write(f"[INFO] Filtered out {filtered_count} rows with missing values")
        # Log filtered rows to file
        log_file = output_dir / "filtered_rows.log"
        output_dir.mkdir(parents=True, exist_ok=True)
        filtered_rows.to_csv(log_file, sep='\t', index=False)
        tqdm.write(f"[INFO] Filtered rows written to: {log_file}")

    if asd_dataset.empty:
        raise ValueError("No valid entries remain after filtering missing values")

    # Prepare PDB directory (download PDB files)
    pdb_ids = asd_dataset['allosteric_pdb'].tolist()
    tqdm.write(f"[INFO] Preparing PDB directory at {output_dir} with {len(pdb_ids)} entries (workers={n_jobs})")
    prepare_pdb_directory(
        pdb_dir=output_dir,
        pdb_ids=pdb_ids,
        clear_existing=clear_existing_pdb,
        n_jobs=n_jobs
    )

    # Extract ligands from PDB files
    ligand_info = pd.DataFrame(
        {
            'pdb_id': asd_dataset['allosteric_pdb'],
            'ligand_chain': asd_dataset['modulator_chain'],
            'ligand_residue': asd_dataset['modulator_resi'],
        }
    )
    tqdm.write(f"[INFO] Extracting ligands (force_ligand_extraction={force_ligand_extraction}, max_diff={max_diff}, workers={n_jobs})")
    prepare_ligands_from_asd(
        pdb_dir=output_dir,
        ligand_info=ligand_info,
        force_ligand_extraction=force_ligand_extraction,
        max_diff=max_diff,
        workers=n_jobs,
        print_summary=True,
        verbose=verbose
    )


def _read_asd_dataset(path: Path) -> pd.DataFrame:
    """
    Read the ASD dataset from a CSV/TSV file.

    Parameters
    ----------
    path : Path
        Path to the ASD dataset file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the ASD dataset.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
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

    Parameters
    ----------
    output_dir : Path
        Directory where split files will be stored.
    raw_dir : Path
        Directory containing raw PDB folders.

    Returns
    -------
    None
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
    "--force-ligand-extraction",
    "-f",
    is_flag=True,
    default=False,
    help="Force re-extraction of ligand files even if they already exist"
)
@click.option(
    "--clear-existing-pdb",
    is_flag=True,
    default=False,
    help="Clear existing PDB files before downloading new ones"
)
@click.option(
    "--max-diff",
    "-m",
    default=0,
    type=int,
    help="Maximum residue ID difference for fuzzy matching (0 = exact match only, 2 = allow ±2)"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output"
)
def main(
        output_dir: Path,
        asd_file: Path,
        jobs: int,
        force_ligand_extraction: bool,
        clear_existing_pdb: bool,
        max_diff: int,
        verbose: bool
):
    """
    Main function to setup data for VN-EGNN allosteric site prediction.

    Parameters
    ----------
    output_dir : Path
        Directory where PDB folders will be stored (one folder per PDB)
    asd_file : Path
        Path to ASD dataset file (CSV/TSV)
    jobs : int
        Number of parallel workers
    force_ligand_extraction : bool
        If True, force re-extraction of ligand files even if they already exist
    clear_existing_pdb : bool
        If True, clear existing PDB files before downloading new ones
    max_diff : int
        Maximum residue ID difference for fuzzy matching (0 = exact match only, 2 = allow ±2)
    verbose : bool
        If True, enable verbose output

    Returns
    -------
    None
    """
    print(f"Loading ASD dataset from: {asd_file}")
    asd_df = _read_asd_dataset(asd_file)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / 'raw'
    setup_data(
        output_dir=raw_dir,
        asd_dataset=asd_df,
        n_jobs=jobs,
        force_ligand_extraction=force_ligand_extraction,
        clear_existing_pdb=clear_existing_pdb,
        max_diff=max_diff,
        verbose=verbose
    )

    split_dir = output_dir / 'splits'
    setup_splits(
        output_dir=split_dir,
        raw_dir=raw_dir
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Command-line tool for extracting binding info from protein ligand complexes.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
import rootutils
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1
from joblib import Parallel, delayed
from rdkit.Chem import Mol
from tqdm.auto import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.protein import read_molecule  # noqa: E402


@dataclass
class ProteinInfo:
    coords: np.ndarray
    res_names: np.ndarray[str]
    atom_names: np.ndarray[str]
    atom_types: np.ndarray[str]
    res_ids: np.ndarray[int]
    chains: np.ndarray[str]


@dataclass
class LigandInfo:
    coords: np.ndarray


def extract_ligand_info(ligand: Mol) -> LigandInfo:
    """Extract ligand information from RDKit molecule."""
    conf = ligand.GetConformer()
    positions = [conf.GetAtomPosition(atom.GetIdx()) for atom in ligand.GetAtoms()]
    coords = np.array(
        [[ligand_point.x, ligand_point.y, ligand_point.z] for ligand_point in positions]
    )
    return LigandInfo(coords=coords)


def extract_protein_info(protein: Structure) -> ProteinInfo:
    """Extract protein information from BioPython structure."""
    coords = []
    res_names = []
    atom_names = []
    atom_types = []
    res_ids = []
    chains = []

    for model in protein:
        for chain in model:
            for residue in chain:
                # Only process standard residues (not HETATM)
                if residue.id[0] != " ":
                    continue
                # Skip non-standard amino acids that seq1() can't convert
                # This ensures consistency with ESM embedding generation
                try:
                    seq1(residue.resname)
                except KeyError:
                    continue
                for atom in residue:
                    coords.append(atom.get_coord())
                    res_names.append(residue.resname)
                    atom_names.append(atom.name)
                    atom_types.append(atom.element)
                    res_ids.append(residue.id[1])
                    chains.append(chain.id)

    return ProteinInfo(
        coords=np.array(coords),
        res_names=np.array(res_names),
        atom_names=np.array(atom_names),
        atom_types=np.array(atom_types),
        res_ids=np.array(res_ids),
        chains=np.array(chains),
    )


def _calculate_depths_internal(protein: Structure) -> Optional[dict]:
    """Internal function to calculate residue depths (called with timeout)."""
    model = protein[0]
    rd = ResidueDepth(model)

    depth_map = {}
    for chain in model:
        for residue in chain:
            res_id = residue.id[1]
            chain_id = chain.id
            key = (chain_id, res_id)

            try:
                depth_tuple = rd[chain_id, residue.id]
                depth_map[key] = depth_tuple[0]
            except KeyError:
                depth_map[key] = np.nan

    return depth_map


def calculate_residue_depths(protein: Structure, timeout: int = 120) -> Optional[dict]:
    """
    Calculate residue depth for all residues in the protein structure.

    Returns a dictionary mapping (chain_id, residue_id) to residue depth,
    or None if MSMS is not available, calculation fails, or times out.

    Args:
        protein: BioPython Structure object
        timeout: Maximum time in seconds to wait for MSMS (default: 120)
    """
    import warnings
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_calculate_depths_internal, protein)
                return future.result(timeout=timeout)
        except FuturesTimeoutError:
            # MSMS took too long
            return None
        except Exception:
            # MSMS might not be installed or other errors
            return None


def extract_binding_site(
    protein_info: ProteinInfo,
    ligand_infos: List[LigandInfo],
    threshold: float = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract binding site information from protein and ligand data."""
    binding_atoms_ligands = []
    binding_site_centers = []

    for ligand_info in ligand_infos:
        dists = np.linalg.norm(
            protein_info.coords[:, None] - ligand_info.coords[None], axis=-1
        )
        below_threshold = (dists <= threshold).any(axis=1)

        binding_atoms_coords = protein_info.coords[below_threshold]
        binding_site_center = binding_atoms_coords.mean(axis=0)

        binding_atoms_ligands.append(below_threshold)
        binding_site_centers.append(binding_site_center)

    return (
        np.array(binding_atoms_ligands).any(axis=0),
        np.array(binding_site_centers),
    )


def _is_valid_binding_file(binding_path: Path) -> bool:
    """
    Check if a binding.npz file exists and is valid.

    Args:
        binding_path: Path to the binding.npz file

    Returns:
        True if the binding file is valid, False otherwise
    """
    if not binding_path.exists():
        return False

    try:
        with np.load(binding_path) as data:
            # Check for required keys
            required_keys = [
                'binding_residues', 'binding_site_centers', 'res_coords',
                'res_names', 'ligand_coords', 'ligand_ids'
            ]
            if not all(key in data.files for key in required_keys):
                return False
            # Check data has valid shape (non-empty)
            if data['res_coords'].shape[0] == 0:
                return False
        return True
    except Exception:
        return False


def process_single_complex(complex_path: Path, threshold: float = 4, skip_depth: bool = False) -> tuple[bool, str]:
    """
    Process a single protein-ligand complex and extract binding information.

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        protein_path = complex_path / "protein.pdb"
        if not protein_path.exists():
            return (False, "protein.pdb not found")

        protein = PDBParser(QUIET=True).get_structure("protein", str(protein_path))
        protein_info = extract_protein_info(protein)

        # Sort so ligand indices (and the site_types below) are deterministic.
        ligand_paths = sorted(
            (complex_path / f for f in os.listdir(complex_path) if f.startswith("ligand")),
            key=lambda p: p.name,
        )
        if not ligand_paths:
            return (False, "No ligand files found")

        ligand_mols = []
        ligand_site_types = []
        failed_ligands = []
        for ligand_path in ligand_paths:
            mol = read_molecule(str(ligand_path))
            if mol is not None:
                ligand_mols.append(mol)
                # Filename convention: `ligand_ortho_*` = orthosteric (0), else
                # allosteric (1). See scripts/allosteric-sites/prepare_orthosteric.py.
                ligand_site_types.append(0 if "ortho" in ligand_path.name.lower() else 1)
            else:
                failed_ligands.append(ligand_path.name)

        if not ligand_mols:
            return (False, f"No valid ligands found (failed: {', '.join(failed_ligands)})")

        ligand_infos = [extract_ligand_info(ligand) for ligand in ligand_mols]

        # Extract binding site information based on atoms being within a threshold
        # distance of a ligand atom.
        binding_atoms, binding_site_centers = extract_binding_site(
            protein_info, ligand_infos, threshold
        )

        # Calculate residue depths (can be skipped if MSMS hangs)
        if skip_depth:
            depth_map = None
        else:
            depth_map = calculate_residue_depths(protein)

        # Extract only CA atoms as used in VN-EGNN as input.
        ca_atoms = protein_info.atom_names == "CA"
        binding_residues = (binding_atoms & ca_atoms)[ca_atoms]

        res_coords = protein_info.coords[ca_atoms]
        res_names = protein_info.res_names[ca_atoms]
        res_ids = protein_info.res_ids[ca_atoms]
        chains = protein_info.chains[ca_atoms]

        # Get residue depths for CA atoms
        if depth_map is not None:
            res_depths = np.array(
                [
                    depth_map.get((chain_id, res_id), np.nan)
                    for chain_id, res_id in zip(chains, res_ids)
                ]
            )
        else:
            # If depth calculation failed, set all to NaN
            res_depths = np.full(len(res_ids), np.nan)

        lig_ids, lig_coords = [], []
        for i, ligand_info in enumerate(ligand_infos):
            lig_ids.extend(len(ligand_info.coords) * [i])
            lig_coords.append(ligand_info.coords)
        lig_ids = np.array(lig_ids)
        lig_coords = np.concatenate(lig_coords, axis=0)

        output_path = complex_path / "binding.npz"
        save_kwargs = dict(
            binding_residues=binding_residues,
            binding_site_centers=binding_site_centers,
            res_coords=res_coords,
            res_names=res_names,
            res_ids=res_ids,
            chains=chains,
            res_depths=res_depths,
            ligand_coords=lig_coords,
            ligand_ids=lig_ids,
        )
        # Only persist per-site labels for mixed proteins (at least one orthosteric
        # ligand). For single-class datasets (sc-PDB, ASD-without-ortho) we omit it so
        # the datamodule's per-dataset `site_type` label applies unchanged.
        site_types = np.array(ligand_site_types, dtype=np.int64)
        if (site_types == 0).any():
            save_kwargs["site_types"] = site_types
        np.savez(str(output_path), **save_kwargs)

        return (True, f"residues={len(res_ids)}, ligands={len(ligand_mols)}")

    except Exception as e:
        return (False, str(e))


@click.command()
@click.option(
    "--path",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to directory containing protein-ligand complex folders",
)
@click.option(
    "--n-jobs",
    "-j",
    default=1,
    type=int,
    help="Number of parallel jobs to run",
)
@click.option(
    "--threshold",
    "-t",
    default=4.0,
    type=float,
    help="Distance threshold for binding site detection (Angstroms)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--backend",
    "-b",
    default="processes",
    type=click.Choice(["threads", "processes"]),
    help="Backend to use for parallel processing",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force regeneration of binding info even if it already exists",
)
@click.option(
    "--skip-depth",
    is_flag=True,
    help="Skip residue depth calculation (MSMS can hang on some structures)",
)
def extract_binding_info(
    path: Path, n_jobs: int, threshold: float, verbose: bool, backend: str, force: bool, skip_depth: bool
):
    """
    Extract binding information from protein-ligand complexes.

    This script processes directories containing protein-ligand complexes,
    extracts binding site information, and saves the results as binding.npz files.

    Each complex directory should contain:
    - protein.pdb: Protein structure file
    - ligand_*.sdf/mol2/pdb: Ligand structure files
    """
    click.echo(f"[INFO] Processing complexes in: {path}")
    click.echo(f"[INFO] Binding site threshold: {threshold} Å")
    if skip_depth:
        click.echo("[INFO] Skipping residue depth calculation (--skip-depth)")

    complex_dirs = [d for d in path.iterdir() if d.is_dir()]
    if not complex_dirs:
        click.echo(f"[WARNING] No directories found in {path}")
        return

    click.echo(f"[INFO] Found {len(complex_dirs)} complex directories")

    # Check for existing binding.npz files (skip unless --force)
    skipped_dirs = []
    if not force:
        dirs_to_process = []
        for complex_dir in complex_dirs:
            binding_file = complex_dir / "binding.npz"
            if binding_file.exists() and _is_valid_binding_file(binding_file):
                skipped_dirs.append(complex_dir)
            else:
                dirs_to_process.append(complex_dir)
        complex_dirs = dirs_to_process

    if not complex_dirs:
        click.echo("[INFO] All complexes already have binding info. Nothing to do.")
        return

    click.echo(f"[INFO] Processing {len(complex_dirs)} complexes")

    # Process complexes and collect results
    all_results = {}
    if n_jobs == 1:
        for complex_dir in tqdm(complex_dirs, desc="Processing complexes"):
            result = process_single_complex(complex_dir, threshold, skip_depth)
            all_results[complex_dir.name] = result
    else:
        # Parallel processing with proper progress tracking
        with tqdm(total=len(complex_dirs), desc="Processing complexes") as pbar:
            results_list = Parallel(n_jobs=n_jobs, prefer=backend, return_as="generator")(
                delayed(process_single_complex)(complex_dir, threshold, skip_depth)
                for complex_dir in complex_dirs
            )
            for complex_dir, result in zip(complex_dirs, results_list):
                all_results[complex_dir.name] = result
                pbar.update(1)

    # Collect failed complexes
    failed_complexes = []
    for complex_name, (success, message) in all_results.items():
        if not success:
            failed_complexes.append((complex_name, message))
            if verbose:
                click.echo(f"[WARNING] {complex_name}: {message}")

    # Count results
    successful_count = sum(1 for _, (success, _) in all_results.items() if success)
    failed_count = len(failed_complexes)

    # Write failed extractions to log file
    if failed_complexes:
        log_file = path / "failed_binding_extractions.log"
        with open(log_file, 'w') as f:
            f.write("complex_name\terror\n")
            for complex_name, error in failed_complexes:
                f.write(f"{complex_name}\t{error}\n")

    # Print summary
    click.echo(f"""
=== Binding Info Extraction Summary ===
Successful: {successful_count}
Skipped (existing): {len(skipped_dirs)}
Failed: {failed_count}
=======================================
""")

    if failed_complexes and not verbose:
        log_file = path / "failed_binding_extractions.log"
        click.echo(f"[INFO] {failed_count} complexes failed. "
                   f"See {log_file} for details or run with -v for verbose output.")

    if failed_complexes:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    extract_binding_info()

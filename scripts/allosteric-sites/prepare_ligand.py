from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


def parse_chain_ids(chain_str: str) -> list[str]:
    """
    Parse chain ID string which may contain multiple chains.
    Handles formats like: "A", "A,B", "A, B", "A;B", etc.
    Returns a list of individual chain IDs.
    """
    if not chain_str or pd.isna(chain_str):
        return []
    # Split by common separators (comma, semicolon, space) but keep single chars
    chain_str = str(chain_str).strip()
    # Split by comma or semicolon, then strip whitespace from each
    chains = re.split(r'[,;]+', chain_str)
    return [c.strip() for c in chains if c.strip()]


class LigandSelect(Select):
    """Select residues matching any of the given chain IDs and residue ID."""

    def __init__(self, chain_ids: list[str], residue_id: int):
        self.chain_ids = set(chain_ids)
        self.residue_id = residue_id

    def accept_residue(self, residue):
        try:
            return (residue.get_parent().id in self.chain_ids and
                    residue.id[1] == self.residue_id)
        except (ValueError, TypeError):
            return False


def diagnose_ligand_extraction(pdb_file: Path, chain_ids: list[str], residue_id: int) -> str:
    """
    Diagnose why a ligand extraction might have failed.
    Returns a diagnostic message.
    """
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    try:
        structure = parser.get_structure("protein", pdb_file)
    except Exception as e:
        return f"Failed to parse PDB: {e}"

    # Check available chains
    available_chains = set()
    for model in structure:
        for chain in model:
            available_chains.add(chain.id)

    # Check if any of the requested chains exist
    missing_chains = [c for c in chain_ids if c not in available_chains]
    if missing_chains:
        return f"Chain(s) {missing_chains} not found. Available: {sorted(available_chains)}"

    # Check available residues in the chains
    available_residues = {}
    for model in structure:
        for chain_id in chain_ids:
            if chain_id in [c.id for c in model]:
                chain = model[chain_id]
                available_residues[chain_id] = set()
                for residue in chain:
                    available_residues[chain_id].add(residue.id[1])

    # Check if residue exists in any of the chains
    found_in_chains = [c for c in chain_ids if residue_id in available_residues.get(c, set())]
    if not found_in_chains:
        # Find closest residue across all chains
        all_residues = set()
        for res_set in available_residues.values():
            all_residues.update(res_set)
        closest = min(all_residues, key=lambda x: abs(x - residue_id), default=None)
        return f"Residue {residue_id} not found in chains {chain_ids}. Closest: {closest}"

    return "Unknown issue - residue found but extraction failed"


def extract_single_ligand(
        pdb_dir: Path,
        pdb_id: str,
        chain_ids_str: str,
        residue_ids_str: str,
        force_ligand_extraction: bool = False
) -> list[tuple[str, Path, str]]:
    """
    Extract ligand(s) from a PDB file and save them.

    The ASD database can specify ligands in two ways:
    1. Single ligand: chain_ids="A", residue_ids="501"
    2. Multiple separate ligands (semicolon-separated): chain_ids="A;B", residue_ids="501;502"
    3. Same ligand across multiple chains (comma-separated): chain_ids="A,B", residue_ids="501"

    :param pdb_dir: Directory containing the PDB files
    :param pdb_id: PDB ID of the protein
    :param chain_ids_str: Chain IDs - semicolon separates different ligands, comma separates chains for same ligand
    :param residue_ids_str: Residue IDs - semicolon separates different ligands
    :param force_ligand_extraction: If True, re-extract even if ligand file exists
    :return: List of tuples[status, ligand_out_file, diagnostic_message]
    """
    protein_dir = pdb_dir / f"{pdb_id}"
    pdb_file = protein_dir / "protein.pdb"

    if not pdb_file.exists():
        return [("missing", pdb_file, "PDB file not found")]

    # Split by semicolon for multiple separate ligands
    chain_groups = str(chain_ids_str).split(";") if chain_ids_str else []
    residue_ids_list = str(residue_ids_str).split(";") if residue_ids_str else []

    # Handle case where there's only one residue ID but multiple chain groups
    if len(residue_ids_list) == 1 and len(chain_groups) > 1:
        residue_ids_list = residue_ids_list * len(chain_groups)

    if len(chain_groups) != len(residue_ids_list):
        raise ValueError(f"Number of chain groups and residue IDs do not match for PDB ID {pdb_id}: "
                        f"chains={chain_groups}, residues={residue_ids_list}")

    results = []
    for i, (chain_group, residue_id_str) in enumerate(zip(chain_groups, residue_ids_list)):
        ligand_out_file = protein_dir / f"ligand_{i}.pdb"

        # Parse the chain group (may contain comma-separated chains like "A,B")
        chain_ids = parse_chain_ids(chain_group)

        # Parse residue ID
        try:
            residue_id = int(residue_id_str.strip())
        except (ValueError, AttributeError):
            results.append(("empty", ligand_out_file, f"Invalid residue ID: {residue_id_str}"))
            continue

        if not force_ligand_extraction and ligand_out_file.exists():
            # Validate existing file has actual atom records
            if _is_valid_ligand_pdb(ligand_out_file):
                results.append(("skipped", ligand_out_file, ""))
                continue
            # If existing file is invalid, re-extract
            ligand_out_file.unlink()

        if not chain_ids:
            results.append(("empty", ligand_out_file, "No valid chain IDs provided"))
            continue

        parser = PDBParser(QUIET=True, PERMISSIVE=True)
        io = PDBIO()
        structure = parser.get_structure(pdb_id, pdb_file)
        io.set_structure(structure)
        io.save(str(ligand_out_file), LigandSelect(chain_ids, residue_id))

        # Validate the extracted ligand file contains actual atom records
        if _is_valid_ligand_pdb(ligand_out_file):
            results.append(("extracted", ligand_out_file, ""))
        else:
            # File is empty or invalid - diagnose and remove it
            diagnostic = diagnose_ligand_extraction(pdb_file, chain_ids, residue_id)
            if ligand_out_file.exists():
                ligand_out_file.unlink()
            results.append(("empty", ligand_out_file, diagnostic))

    return results


def _is_valid_ligand_pdb(pdb_file: Path) -> bool:
    """
    Check if a PDB file contains valid atom records.
    A valid ligand PDB should have at least one ATOM or HETATM record.
    """
    if not pdb_file.exists():
        return False

    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    return True
        return False
    except Exception:
        return False


def prepare_ligands_from_asd(
        pdb_dir: Path,
        ligand_info: pd.DataFrame,
        force_ligand_extraction: bool = False,
        workers: int = 8,
        print_summary: bool = True,
        verbose: bool = False
) -> None:
    """
    Save ligand structures from ASD dataset PDB files.
    :param pdb_dir: Directory containing PDB files organized by PDB ID
    :param ligand_info: DataFrame with columns ['pdb_id', 'ligand_chain', 'ligand_residue']
    :param force_ligand_extraction: If True, re-extract ligands even if they already exist
    :param workers: Number of parallel workers for extraction
    :param print_summary: If True, print a summary of extraction results
    :param verbose: If True, print detailed diagnostic messages for failed extractions
    :return: None
    """
    tqdm.write("[INFO] Preparing ligands from ASD dataset...")
    counts = {"extracted": 0, "skipped": 0, "missing": 0, "empty": 0, "errors": 0}
    failed_extractions = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                extract_single_ligand,
                pdb_dir,
                row['pdb_id'],
                row['ligand_chain'],
                row['ligand_residue'],
                force_ligand_extraction
            ): (row['pdb_id'], row['ligand_chain'], row['ligand_residue'])
            for _, row in ligand_info.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting ligands"):
            pdb_id, chain_id, residue_id = futures[future]
            try:
                results = future.result()
                for status, path, diagnostic in results:
                    counts[status] += 1
                    if status == "missing":
                        tqdm.write(f"[WARNING] PDB file missing for {pdb_id}")
                    elif status == "empty":
                        msg = f"{pdb_id} chain {chain_id} residue {residue_id}"
                        if verbose and diagnostic:
                            tqdm.write(f"[WARNING] Ligand extraction failed: {msg} - {diagnostic}")
                        failed_extractions.append((pdb_id, chain_id, residue_id, diagnostic))
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to extract ligand for {pdb_id} chain {chain_id}: {e}")

                counts["errors"] += 1

    tqdm.write("[INFO] Ligand preparation complete.")
    if print_summary:
        tqdm.write(f""" 
=== Ligand extraction summary ===
Extracted: {counts['extracted']}
Skipped (existing): {counts['skipped']}
Missing PDB files: {counts['missing']}
Empty (ligand not found): {counts['empty']}
Errors: {counts['errors']}
===============================
""")

        if failed_extractions and not verbose:
            tqdm.write(f"[INFO] {len(failed_extractions)} ligands could not be extracted. "
                      f"Run with verbose=True for details.")

    # Write failed extractions to a log file for debugging
    if failed_extractions:
        log_file = pdb_dir / "failed_ligand_extractions.log"
        with open(log_file, 'w') as f:
            f.write("pdb_id\tchain\tresidue\tdiagnostic\n")
            for pdb_id, chain, residue, diag in failed_extractions:
                f.write(f"{pdb_id}\t{chain}\t{residue}\t{diag}\n")
        tqdm.write(f"[INFO] Failed extraction details written to: {log_file}")

    return

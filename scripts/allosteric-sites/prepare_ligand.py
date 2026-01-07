from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


def parse_chain_ids(chain_str: str) -> list[str]:
    """
    Parse chain ID string which may contain multiple chains.
    Handles formats like: "A", "A,B", "A, B", "A;B", etc.

    Parameters
    ----------
    chain_str : str
        The chain ID string to parse.

    Returns
    -------
    list[str]
        A list of chain IDs.
    """
    if not chain_str or pd.isna(chain_str):
        return []
    # Split by common separators (comma, semicolon, space) but keep single chars
    chain_str = str(chain_str).strip()
    # Split by comma or semicolon, then strip whitespace from each
    chains = re.split(r'[,;]+', chain_str)
    return [c.strip() for c in chains if c.strip()]


def parse_residue_id(residue_str: str) -> tuple[list[int], str | None]:
    """
    Parse residue ID string which may contain various formats.

    Supported formats:
    - Single: "501" -> ([501], None)
    - Multiple comma: "401,402" -> ([401, 402], None)
    - Multiple slash: "1585/1586" -> ([1585, 1586], None)
    - Range: "1-141" -> ([], "range format - peptide ligand")

    Parameters
    ----------
    residue_str : str
        The residue ID string to parse.

    Returns
    -------
    tuple[list[int], str | None]
        A tuple containing a list of residue IDs and an optional error message.
    """
    if not residue_str or pd.isna(residue_str):
        return [], "Empty residue ID"

    residue_str = str(residue_str).strip()

    # Check for range format (e.g., "1-141") - these are peptide ligands, skip them
    if re.match(r'^\d+-\d+$', residue_str):
        return [], f"Range format '{residue_str}' indicates peptide ligand - skipping"

    # Handle multiple residues separated by comma or slash
    if ',' in residue_str or '/' in residue_str:
        parts = re.split(r'[,/]+', residue_str)
        residue_ids = []
        for part in parts:
            part = part.strip()
            try:
                residue_ids.append(int(part))
            except ValueError:
                return [], f"Invalid residue ID component: {part}"
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for rid in residue_ids:
            if rid not in seen:
                seen.add(rid)
                unique_ids.append(rid)
        return unique_ids, None

    # Single residue ID
    try:
        return [int(residue_str)], None
    except ValueError:
        return [], f"Invalid residue ID: {residue_str}"


def extract_ligand_with_conect(
        pdb_file: Path,
        chain_ids: list[str],
        residue_id: int,
        output_file: Path
) -> bool:
    """
    Extract ligand HETATM records along with corresponding CONECT records.

    This ensures proper bond connectivity is preserved for RDKit parsing.

    Parameters
    ----------
    pdb_file : Path
        Path to the source PDB file.
    chain_ids : list[str]
        List of chain IDs to search within.
    residue_id : int
        The residue ID of the ligand to extract.
    output_file : Path
        Path to write the extracted ligand PDB.

    Returns
    -------
    bool
        True if ligand was extracted successfully, False otherwise.
    """
    hetatm_lines = []
    conect_lines = []
    atom_serial_set = set()

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                # PDB format: columns 22 = chain ID, 23-26 = residue sequence number
                chain = line[21].strip()
                try:
                    res_id = int(line[22:26].strip())
                except ValueError:
                    continue
                if chain in chain_ids and res_id == residue_id:
                    hetatm_lines.append(line)
                    # Track atom serial numbers (columns 7-11)
                    try:
                        atom_serial_set.add(int(line[6:11].strip()))
                    except ValueError:
                        pass
            elif line.startswith('CONECT'):
                conect_lines.append(line)

    if not hetatm_lines:
        return False

    # Filter CONECT lines to only include atoms in our ligand
    filtered_conect = []
    for line in conect_lines:
        # CONECT format: CONECT serial1 serial2 serial3 ...
        # Each serial is 5 characters starting at position 6
        try:
            primary_serial = int(line[6:11].strip())
            if primary_serial in atom_serial_set:
                # Keep only connections to atoms also in our ligand
                # Parse all connected atoms and filter
                new_line = line[:11]  # Keep "CONECT" + primary serial
                pos = 11
                while pos + 5 <= len(line.rstrip()):
                    try:
                        connected_serial = int(line[pos:pos+5].strip())
                        if connected_serial in atom_serial_set:
                            new_line += f"{connected_serial:>5}"
                    except ValueError:
                        pass
                    pos += 5
                if len(new_line) > 11:  # Has at least one valid connection
                    filtered_conect.append(new_line + '\n')
        except (ValueError, IndexError):
            continue

    with open(output_file, 'w') as f:
        for line in hetatm_lines:
            f.write(line)
        for line in filtered_conect:
            f.write(line)
        f.write('END\n')

    return True


def find_closest_residue(
        pdb_file: Path,
        chain_ids: list[str],
        target_residue: int,
        max_diff: int = 2
) -> int | None:
    """
    Find the closest matching residue ID within max_diff of the target.
    Only returns a match if it's a HETATM residue (ligand).

    Parameters
    ----------
    pdb_file : Path
        Path to the PDB file.
    chain_ids : list[str]
        List of chain IDs to search within.
    target_residue : int
        The target residue ID to match.
    max_diff : int
        Maximum allowed difference from the target residue ID.

    Returns
    -------
    int | None
        The closest matching residue ID, or None if no match found.
    """
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    try:
        structure = parser.get_structure("protein", pdb_file)
    except Exception:
        return None

    candidates = []
    for model in structure:
        for chain_id in chain_ids:
            if chain_id not in [c.id for c in model]:
                continue
            chain = model[chain_id]
            for residue in chain:
                res_id = residue.id[1]
                hetflag = residue.id[0]
                # Only consider HETATM residues (ligands) that are close to target
                if hetflag.strip() and abs(res_id - target_residue) <= max_diff:
                    candidates.append(res_id)

    if not candidates:
        return None

    # Return the closest match
    return min(candidates, key=lambda x: abs(x - target_residue))


def diagnose_ligand_extraction(
        pdb_file: Path,
        chain_ids: list[str],
        residue_id: int
) -> str:
    """
    Diagnose why a ligand extraction might have failed.

    Parameters
    ----------
    pdb_file : Path
        Path to the PDB file.
    chain_ids : list[str]
        List of chain IDs to check.
    residue_id : int
        The residue ID that was attempted to be extracted.

    Returns
    -------
    str
        Diagnostic message explaining the failure reason.
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

    # Check available HETATM residues (ligands) in the chains
    hetatm_residues = {}
    all_residues = {}
    for model in structure:
        for chain_id in chain_ids:
            if chain_id in [c.id for c in model]:
                chain = model[chain_id]
                hetatm_residues[chain_id] = set()
                all_residues[chain_id] = set()
                for residue in chain:
                    res_id = residue.id[1]
                    hetflag = residue.id[0]
                    all_residues[chain_id].add(res_id)
                    # HETATM residues have non-empty hetflag
                    if hetflag.strip():
                        hetatm_residues[chain_id].add(res_id)

    # Check if the exact residue exists as HETATM
    found_hetatm = [c for c in chain_ids if residue_id in hetatm_residues.get(c, set())]
    if found_hetatm:
        return "Unknown issue - HETATM residue found but extraction failed"

    # Check if the exact residue exists at all (as standard residue)
    found_any = [c for c in chain_ids if residue_id in all_residues.get(c, set())]
    if found_any:
        return f"Residue {residue_id} exists in chains {found_any} but is not a HETATM (ligand)"

    # Find closest HETATM residue across all chains
    all_hetatm = set()
    for res_set in hetatm_residues.values():
        all_hetatm.update(res_set)

    # Find closest standard residue across all chains
    all_std = set()
    for res_set in all_residues.values():
        all_std.update(res_set)

    closest_hetatm = min(all_hetatm, key=lambda x: abs(x - residue_id), default=None) if all_hetatm else None
    closest_any = min(all_std, key=lambda x: abs(x - residue_id), default=None) if all_std else None

    if closest_hetatm is not None:
        diff = abs(closest_hetatm - residue_id)
        return f"Residue {residue_id} not found in chains {chain_ids}. Closest HETATM: {closest_hetatm} (diff={diff})"
    elif closest_any is not None:
        diff = abs(closest_any - residue_id)
        return f"Residue {residue_id} not found in chains {chain_ids}. No HETATM residues nearby. Closest residue: {closest_any} (diff={diff})"
    else:
        return f"Residue {residue_id} not found in chains {chain_ids}. No residues found."


def extract_single_ligand(
        pdb_dir: Path,
        pdb_id: str,
        chain_ids_str: str,
        residue_ids_str: str,
        force_ligand_extraction: bool = False,
        max_diff: int = 0
) -> list[tuple[str, Path, str]]:
    """
    Extract ligand(s) from a PDB file and save them.

    The ASD database can specify ligands in various formats:
    1. Single ligand: chain_ids="A", residue_ids="501"
    2. Multiple separate ligands (semicolon-separated): chain_ids="A;B", residue_ids="501;502"
    3. Same ligand across multiple chains (comma-separated): chain_ids="A,B", residue_ids="501"
    4. Multiple residues (comma or slash): residue_ids="401,402" or "1585/1586"

    Parameters
    ----------
    pdb_dir : Path
        Directory containing PDB files organized by PDB ID.
    pdb_id : str
        The PDB ID of the protein.
    chain_ids_str : str
        Chain ID(s) string from the dataset.
    residue_ids_str : str
        Residue ID(s) string from the dataset.
    force_ligand_extraction : bool
        If True, re-extract ligands even if they already exist.
    max_diff : int
        Maximum residue ID difference for fuzzy matching (0 = exact match only).

    Returns
    -------
    list[tuple[str, Path, str]]
        A list of tuples with extraction status, output file path, and diagnostic message.
    """
    protein_dir = pdb_dir / f"{pdb_id}"
    pdb_file = protein_dir / "protein.pdb"

    if not pdb_file.exists():
        return [("missing", pdb_file, "PDB file not found")]

    # Split by semicolon for multiple separate ligand entries
    chain_groups = str(chain_ids_str).split(";") if chain_ids_str else []
    residue_groups = str(residue_ids_str).split(";") if residue_ids_str else []

    # Handle case where there's only one residue group but multiple chain groups
    if len(residue_groups) == 1 and len(chain_groups) > 1:
        residue_groups = residue_groups * len(chain_groups)

    if len(chain_groups) != len(residue_groups):
        return [("empty", protein_dir / "ligand_0.pdb",
                 f"Chain/residue count mismatch: {len(chain_groups)} chains, {len(residue_groups)} residues")]

    results = []
    ligand_idx = 0

    for chain_group, residue_group in zip(chain_groups, residue_groups):
        # Parse chain IDs (may contain comma-separated chains like "A,B")
        chain_ids = parse_chain_ids(chain_group)

        if not chain_ids:
            results.append(("empty", protein_dir / f"ligand_{ligand_idx}.pdb", "No valid chain IDs provided"))
            ligand_idx += 1
            continue

        # Parse residue IDs (may be single, comma-separated, slash-separated, or range)
        residue_ids, error = parse_residue_id(residue_group)

        if error:
            results.append(("empty", protein_dir / f"ligand_{ligand_idx}.pdb", error))
            ligand_idx += 1
            continue

        # Extract each residue as a separate ligand file
        for residue_id in residue_ids:
            ligand_out_file = protein_dir / f"ligand_{ligand_idx}.pdb"
            ligand_idx += 1

            if not force_ligand_extraction and ligand_out_file.exists():
                if _is_valid_ligand_pdb(ligand_out_file):
                    results.append(("skipped", ligand_out_file, ""))
                    continue
                ligand_out_file.unlink()

            # Try to extract with exact residue ID using CONECT-preserving extraction
            extracted = extract_ligand_with_conect(pdb_file, chain_ids, residue_id, ligand_out_file)

            if extracted and _is_valid_ligand_pdb(ligand_out_file):
                results.append(("extracted", ligand_out_file, ""))
            elif max_diff > 0:
                # Try fuzzy matching - find closest HETATM residue within ±max_diff
                closest = find_closest_residue(pdb_file, chain_ids, residue_id, max_diff=max_diff)
                if closest and closest != residue_id:
                    # Re-extract with corrected residue ID
                    extracted = extract_ligand_with_conect(pdb_file, chain_ids, closest, ligand_out_file)
                    if extracted and _is_valid_ligand_pdb(ligand_out_file):
                        results.append(("extracted", ligand_out_file,
                                        f"Fuzzy matched: {residue_id} -> {closest}"))
                        continue

                # Fuzzy match failed - diagnose and report
                if ligand_out_file.exists():
                    ligand_out_file.unlink()
                diagnostic = diagnose_ligand_extraction(pdb_file, chain_ids, residue_id)
                results.append(("empty", ligand_out_file, diagnostic))
            else:
                # No fuzzy matching - diagnose and report
                if ligand_out_file.exists():
                    ligand_out_file.unlink()
                diagnostic = diagnose_ligand_extraction(pdb_file, chain_ids, residue_id)
                results.append(("empty", ligand_out_file, diagnostic))

    return results


def _is_valid_ligand_pdb(pdb_file: Path) -> bool:
    """
    Check if a PDB file contains valid atom records.
    A valid ligand PDB should have at least one ATOM or HETATM record.

    Parameters
    ----------
    pdb_file : Path
        Path to the PDB file to check.

    Returns
    -------
    bool
        True if the PDB file contains ATOM or HETATM records, False otherwise.
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
        max_diff: int = 0,
        workers: int = 8,
        print_summary: bool = True,
        verbose: bool = False
) -> None:
    """
    Save ligand structures from ASD dataset PDB files.

    Parameters
    ----------
    pdb_dir : Path
        Directory containing PDB files organized by PDB ID.
    ligand_info : pd.DataFrame
        DataFrame with columns: 'pdb_id', 'ligand_chain', 'ligand_residue'.
    force_ligand_extraction : bool
        If True, re-extract ligands even if they already exist.
    max_diff : int
        Maximum residue ID difference for fuzzy matching (0 = exact match only).
    workers : int
        Number of parallel worker threads to use.
    print_summary : bool
        If True, print a summary of the extraction results.
    verbose : bool
        If True, print detailed warnings for failed extractions.

    Returns
    -------
    None
    """
    tqdm.write("[INFO] Preparing ligands from ASD dataset...")
    counts = {"extracted": 0, "skipped": 0, "missing": 0, "empty": 0, "errors": 0}
    failed_extractions = []

    # Group ligands by PDB ID to avoid race conditions when multiple rows have the same PDB
    # and to correctly number ligands across all entries for the same PDB
    grouped = ligand_info.groupby('pdb_id').agg({
        'ligand_chain': lambda x: ';'.join(str(v) for v in x),
        'ligand_residue': lambda x: ';'.join(str(v) for v in x)
    }).reset_index()

    # Count total ligands after grouping (each semicolon-separated value is a ligand)
    total_ligands = sum(
        len(str(row['ligand_residue']).split(';'))
        for _, row in grouped.iterrows()
    )

    tqdm.write(
        f"[INFO] Found {len(ligand_info)} dataset entries across {len(grouped)} unique PDB IDs ({total_ligands} total ligands)")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                extract_single_ligand,
                pdb_dir,
                row['pdb_id'],
                row['ligand_chain'],
                row['ligand_residue'],
                force_ligand_extraction,
                max_diff
            ): (row['pdb_id'], row['ligand_chain'], row['ligand_residue'])
            for _, row in grouped.iterrows()
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

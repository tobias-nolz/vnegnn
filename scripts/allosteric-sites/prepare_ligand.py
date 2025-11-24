from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class LigandSelect(Select):
    def __init__(self, chain_id: str, residue_id: str):
        self.chain_id = chain_id
        self.residue_id = residue_id

    def accept_residue(self, residue):
        return (residue.get_parent().id == self.chain_id and
                residue.id[1] == int(self.residue_id))


def extract_single_ligand(
        pdb_dir: Path,
        pdb_id: str,
        chain_ids: str,
        residue_ids: str,
        skip_existing: bool = True
) -> list[tuple[str, Path]]:
    """
    Extract a single ligand from a PDB file and save it.
    :param pdb_dir: Directory containing the PDB files
    :param pdb_id: PDB ID of the protein
    :param chain_ids: Chain IDs of the ligand
    :param residue_ids: Residue IDs of the ligand
    :param skip_existing: If True, skip extraction if ligand file already exists
    :return: List of tuples[status, ligand_out_file]
        status: "extracted", "skipped", or "missing"
        ligand_out_file: Path to the extracted ligand file
    """
    protein_dir = pdb_dir / f"{pdb_id}"
    pdb_file = protein_dir / "protein.pdb"

    if not pdb_file.exists():
        return [("missing", pdb_file)]

    chain_ids = str(chain_ids).split(";")
    residue_ids = str(residue_ids).split(";")

    if len(chain_ids) != len(residue_ids):
        raise ValueError(f"Number of chain IDs and residue IDs do not match for PDB ID {pdb_id}")

    results = []
    for i, (chain_id, residue_ids) in enumerate(zip(chain_ids, residue_ids)):
        ligand_out_file = protein_dir / f"ligand_{i}.pdb"

        if skip_existing and ligand_out_file.exists():
            results.append(("skipped", ligand_out_file))
            continue

        parser = PDBParser(QUIET=True, PERMISSIVE=True)
        io = PDBIO()
        structure = parser.get_structure(pdb_id, pdb_file)
        io.set_structure(structure)
        io.save(str(ligand_out_file), LigandSelect(chain_id, residue_ids))
        results.append(("extracted", ligand_out_file))

    return results


def prepare_ligands_from_asd(
        pdb_dir: Path,
        ligand_info: pd.DataFrame,
        skip_existing: bool = True,
        workers: int = 8,
        print_summary: bool = True
) -> None:
    """
    Save ligand structures from ASD dataset PDB files.
    :param pdb_dir: Directory containing PDB files organized by PDB ID
    :param ligand_info: DataFrame with columns ['pdb_id', 'ligand_chain', 'ligand_residue']
    :param skip_existing: If True, skip extraction if ligand file already exists
    :param workers: Number of parallel workers (not used in this implementation)
    :param print_summary: If True, print a summary of extraction results
    :return: None
    """
    tqdm.write("[INFO] Preparing ligands from ASD dataset...")
    counts = {"extracted": 0, "skipped": 0, "missing": 0}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                extract_single_ligand,
                pdb_dir,
                row['pdb_id'],
                row['ligand_chain'],
                row['ligand_residue'],
                skip_existing
            ): (row['pdb_id'], row['ligand_chain'])
            for _, row in ligand_info.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting ligands"):
            pdb_id, chain_id = futures[future]
            try:
                results = future.result()
                for status, _ in results:
                    counts[status] += 1
                if status == "missing":
                    tqdm.write(f"[WARNING] PDB file missing for {pdb_id}")
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to extract ligand for {pdb_id} chain {chain_id}: {e}")

    tqdm.write("[INFO] Ligand preparation complete.")
    if print_summary:
        tqdm.write(f""" 
=== Ligand extraction summary ===
Extracted: {counts['extracted']}
Skipped (existing): {counts['skipped']}
Missing PDB files: {counts['missing']}
===============================
""")
    return

"""
Extract *orthosteric-candidate* ligands from ASD protein structures.

ASD annotates only the ALLOSTERIC modulator of each protein. To obtain orthosteric
(class 0) sites within the same protein -- so the classifier sees both classes on one
structure -- we harvest the *other* co-crystallized ligands from `protein.pdb`.

IMPORTANT CAVEAT: "a co-crystallized ligand that is not the annotated allosteric
modulator" is only a *heuristic* for "orthosteric". Such a ligand could instead be a
cofactor, a crystallization additive, a metal, or even a second allosteric site. We
reduce false positives with (a) a blocklist of common non-ligand HETATMs (water, ions,
buffers, cryoprotectants) and (b) a minimum heavy-atom count, but the labels remain
noisy. Treat orthosteric-augmented ASD data as a training aid, not gold-standard truth,
and prefer a principled active-site source (UniProt / M-CSA) if available.

Output: `ligand_ortho_<i>.pdb` files next to the existing `ligand_<i>.pdb` (allosteric)
files. Downstream, scripts/extract_binding_info.py labels any `ligand_ortho_*` site as
orthosteric (0) and the rest as allosteric (1).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from prepare_ligand import extract_ligand_with_conect, parse_chain_ids, parse_residue_id

# HETATM residue names that are almost never the biologically relevant ligand:
# water, common ions, and frequent crystallization/cryo additives.
EXCLUDED_HETATMS = {
    "HOH", "DOD", "WAT",  # water
    "NA", "K", "CL", "MG", "CA", "ZN", "MN", "FE", "FE2", "CU", "CU1", "NI", "CO",
    "CD", "HG", "BA", "SR", "CS", "RB", "LI", "AL", "IOD", "BR", "F",  # ions
    "SO4", "PO4", "NO3", "ACT", "EDO", "GOL", "PEG", "PG4", "PGE", "1PE",
    "MPD", "DMS", "TRS", "EPE", "MES", "BME", "IPA", "FMT", "ACY", "TLA", "CIT",
    "MRD", "BOG", "DTT", "GSH", "SPD", "SPM", "IMD", "AZI", "CAC", "PO3", "SCN",
    "FLC", "P6G", "2PE", "12P", "15P", "PE4", "PEO", "OGA",  # buffers/cryo
}


def _residue_heavy_atom_count(pdb_file: Path, chain: str, resseq: int, resname: str) -> int:
    count = 0
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            if line[21].strip() != chain:
                continue
            try:
                if int(line[22:26]) != resseq:
                    continue
            except ValueError:
                continue
            if line[17:20].strip() != resname:
                continue
            element = line[76:78].strip() if len(line) >= 78 else line[12:14].strip()
            if element.upper() != "H":
                count += 1
    return count


def discover_orthosteric_ligands(
    pdb_file: Path,
    exclude_resids: set[tuple[str, int]],
    min_heavy_atoms: int = 6,
) -> list[tuple[str, int, str]]:
    """Return (chain, resseq, resname) of candidate orthosteric HETATM ligands.

    Excludes the given (chain, resseq) pairs (the allosteric modulator residues),
    blocklisted residue names, and anything below ``min_heavy_atoms`` heavy atoms.
    """
    seen: dict[tuple[str, int], str] = {}
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            chain = line[21].strip()
            resname = line[17:20].strip()
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            key = (chain, resseq)
            if key in seen:
                continue
            if resname in EXCLUDED_HETATMS:
                continue
            if key in exclude_resids:
                continue
            seen[key] = resname

    candidates = []
    for (chain, resseq), resname in seen.items():
        if _residue_heavy_atom_count(pdb_file, chain, resseq, resname) >= min_heavy_atoms:
            candidates.append((chain, resseq, resname))
    return candidates


def extract_orthosteric_for_protein(
    protein_dir: Path,
    modulator_chain_str: str,
    modulator_residue_str: str,
    min_heavy_atoms: int = 6,
    force: bool = False,
) -> tuple[str, int]:
    """Extract orthosteric-candidate ligands for one protein folder.

    Returns (pdb_id, num_extracted).
    """
    pdb_file = protein_dir / "protein.pdb"
    if not pdb_file.exists():
        return (protein_dir.name, 0)

    # Parse the allosteric modulator (chain, resseq) pairs so we never relabel them.
    exclude_resids: set[tuple[str, int]] = set()
    chain_groups = [c.strip() for c in str(modulator_chain_str).split(";") if c.strip()]
    residue_groups = [r.strip() for r in str(modulator_residue_str).split(";") if r.strip()]
    if len(residue_groups) == 1 and len(chain_groups) > 1:
        residue_groups = residue_groups * len(chain_groups)
    if len(chain_groups) == 1 and len(residue_groups) > 1:
        chain_groups = chain_groups * len(residue_groups)
    for chain_group, residue_group in zip(chain_groups, residue_groups):
        chains = parse_chain_ids(chain_group)
        resids, err = parse_residue_id(residue_group)
        if err:
            continue
        for chain in chains:
            for resid in resids:
                exclude_resids.add((chain, resid))

    candidates = discover_orthosteric_ligands(pdb_file, exclude_resids, min_heavy_atoms)

    # Remove stale ortho ligands when forcing a rebuild.
    if force:
        for old in protein_dir.glob("ligand_ortho_*.pdb"):
            old.unlink()

    extracted = 0
    for i, (chain, resseq, _resname) in enumerate(candidates):
        out_file = protein_dir / f"ligand_ortho_{i}.pdb"
        if out_file.exists() and not force:
            extracted += 1
            continue
        if extract_ligand_with_conect(pdb_file, [chain], resseq, out_file):
            extracted += 1
        elif out_file.exists():
            out_file.unlink()
    return (protein_dir.name, extracted)


def prepare_orthosteric_from_asd(
    pdb_dir: Path,
    ligand_info: pd.DataFrame,
    min_heavy_atoms: int = 6,
    force: bool = False,
    workers: int = 8,
    print_summary: bool = True,
) -> None:
    """Extract orthosteric-candidate ligands for every ASD protein in ``ligand_info``.

    ``ligand_info`` must have columns 'pdb_id', 'ligand_chain', 'ligand_residue'
    (the allosteric modulator annotation), as built in setup_data.py.
    """
    tqdm.write("[INFO] Extracting orthosteric-candidate ligands (heuristic)...")
    grouped = ligand_info.groupby("pdb_id").agg({
        "ligand_chain": lambda x: ";".join(str(v) for v in x),
        "ligand_residue": lambda x: ";".join(str(v) for v in x),
    }).reset_index()

    total_extracted = 0
    proteins_with_ortho = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                extract_orthosteric_for_protein,
                pdb_dir / row["pdb_id"],
                row["ligand_chain"],
                row["ligand_residue"],
                min_heavy_atoms,
                force,
            ): row["pdb_id"]
            for _, row in grouped.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Orthosteric"):
            try:
                _pdb_id, n = future.result()
                total_extracted += n
                proteins_with_ortho += int(n > 0)
            except Exception as e:  # noqa: BLE001
                tqdm.write(f"[ERROR] Orthosteric extraction failed for {futures[future]}: {e}")

    if print_summary:
        tqdm.write(f"""
=== Orthosteric extraction summary ===
Proteins with >=1 orthosteric candidate: {proteins_with_ortho}/{len(grouped)}
Total orthosteric ligands extracted:     {total_extracted}
(heuristic: non-modulator drug-like HETATMs, >= {min_heavy_atoms} heavy atoms)
======================================
""")

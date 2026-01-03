from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import os
import re
import time
import requests


def normalize_pdb_tokens(pbd_id: str) -> list[str]:
    """
    Normalize and extract PDB IDs from a raw string.
    Handles cases like:
    - Single ID: "1ABC"
    - Multiple IDs: "1ABC; 2DEF"
    - IDs with chain: "1ABC_A"

    Parameters
    ----------
    pbd_id : str
        Raw PDB ID string.

    Returns
    -------
    list[str]
        List of normalized PDB IDs (uppercase, 4-char).
    """
    s = pbd_id.strip()

    # split by common separators first
    parts = re.split(r'[;,/|\s]+', s)
    ids = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # look for the first alphanumeric run of length=4 (PDB ids are 4 chars)
        m = re.search(r'([A-Za-z0-9]{4})', p)
        if m:
            ids.append(m.group(1).upper())
    return ids


def download_single(
        pdb_id: str,
        out_dir: Path,
        max_retries=3,
        timeout=25) -> tuple[str, str, str | None]:
    """
    Download a single PDB ID and save to out_dir.
    If the file already exists, skip download.

    Parameters
    ----------
    pdb_id : str
        PDB ID to download.
    out_dir : Path
        Directory to save the PDB file.
    max_retries : int
        Maximum number of download attempts.
    timeout : int
        Timeout for each download attempt in seconds.

    Returns
    -------
    tuple[str, str, str | None]
        (pdb_id, status, file_path)
        status: "downloaded", "exists", or "not_found"
        file_path: path to the downloaded file or None if not found
    """
    pdb_id_up = pdb_id.upper()
    out_dir = os.path.join(out_dir, f"{pdb_id_up}")
    if os.path.exists(out_dir):
        return pdb_id_up, "exists", None

    os.makedirs(out_dir, exist_ok=True)
    out_pdb = os.path.join(out_dir, "protein.pdb")

    url = f"https://files.rcsb.org/download/{pdb_id_up}.pdb"
    attempt = 0

    while attempt < max_retries:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.content and len(r.content) > 100:
                with open(out_pdb, "wb") as fh:
                    fh.write(r.content)
                return pdb_id_up, "downloaded", out_pdb
            else:
                # 404 or small content -> return not found
                return pdb_id_up, "not_found", None
        except requests.RequestException as e:
            attempt += 1
            time.sleep(1 + attempt * 0.5)

    return pdb_id_up, "not_found", None


def prepare_pdb_directory(
        pdb_dir: Path,
        pdb_ids: list[str],
        clear_existing: bool = False,
        n_jobs: int = 8,
        print_summary: bool = True) -> None:
    """
    Prepare a directory with PDB files for the given PDB IDs.
    Downloads files in parallel using multiple threads.

    Parameters
    ----------
    pdb_dir : Path
        Directory to save PDB files.
    pdb_ids : list[str]
        List of raw PDB ID strings.
    clear_existing : bool
        If True, clear existing PDB files in the directory before downloading.
    n_jobs : int
        Number of parallel download threads.
    print_summary : bool
        If True, print a summary of the download results.

    Returns
    -------
    None
    """
    pdb_ids = sorted({pid for raw in pdb_ids if raw for pid in normalize_pdb_tokens(raw)})
    pdb_dir.mkdir(parents=True, exist_ok=True)

    if clear_existing:
        for pdb_file in pdb_dir.rglob("*.pdb"):
            pdb_file.unlink()

    tqdm.write(f"[INFO] Starting download of {len(pdb_ids)} PDB files to {pdb_dir} using {n_jobs} workers...")
    counts = {"downloaded": 0, "exists": 0, "not_found": 0}
    failed_downloads = []

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futures = {
            ex.submit(
                download_single,
                pid,
                pdb_dir
            ): pid
            for pid in pdb_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading PDB files", unit="pdb"):
            pid = futures[future]
            try:
                _, status, _ = future.result()
                counts[status] += 1
                if status == "not_found":
                    tqdm.write(f"[WARNING] {pid} not found on RCSB")
                    failed_downloads.append(pid)
            except Exception as e:
                tqdm.write(f"[ERROR] {pid} -> {e}")
                failed_downloads.append(pid)

    tqdm.write("[INFO] PDB download complete.")
    if print_summary:
        tqdm.write(f"""
=== PDB processing summary ===
Downloaded: {counts.get('downloaded', 0)}
Already existed: {counts.get('exists', 0)}
Not found (404/empty): {counts.get('not_found', 0)}
Saved to: {pdb_dir.resolve()}
==============================
""")

    # Write failed downloads to a log file
    if failed_downloads:
        log_file = pdb_dir / "failed_pdb_downloads.log"
        with open(log_file, 'w') as f:
            f.write("pdb_id\n")
            for pid in sorted(failed_downloads):
                f.write(f"{pid}\n")
        tqdm.write(f"[INFO] Failed downloads written to: {log_file}")

    return

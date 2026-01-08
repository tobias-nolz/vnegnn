#!/usr/bin/env python3
"""
Command-line tool for generating ESM embeddings from protein PDB files.
"""

import os
from pathlib import Path

import click
import esm
import h5py
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def extract_sequence_from_pdb(pdb_path: Path) -> tuple[str, list[tuple[str, str]]]:
    """
    Extract amino acid sequence from PDB file.

    Returns:
        tuple: (concatenated_sequence, list of (chain_id, chain_sequence) tuples)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    chain_sequences = []
    full_sequence = ""
    skipped_residues = []

    for model in structure:
        for chain in model:
            chain_seq = ""
            for residue in chain:
                if residue.id[0] == " ":  # Only standard residues
                    try:
                        aa_3letter = residue.resname
                        aa_1letter = seq1(aa_3letter)

                        # Skip residues without CA atoms
                        # (must match binding info extraction)
                        if "CA" not in residue:
                            skipped_residues.append(
                                f"{chain.id}:{residue.resname}{residue.id[1]}"
                                f" (no CA)"
                            )
                            continue

                        chain_seq += aa_1letter
                    except KeyError:
                        # Non-standard amino acid that seq1() doesn't recognize
                        skipped_residues.append(
                            f"{chain.id}:{residue.resname}{residue.id[1]}"
                        )
                        continue

            if chain_seq:  # Only add non-empty chains
                chain_id = chain.id if chain.id.strip() else "A"
                chain_sequences.append((chain_id, chain_seq))
                full_sequence += chain_seq

    if skipped_residues:
        import warnings

        residue_list = ", ".join(skipped_residues[:5])
        extra_msg = (
            f" and {len(skipped_residues) - 5} more"
            if len(skipped_residues) > 5
            else ""
        )
        warnings.warn(
            f"Skipped {len(skipped_residues)} non-standard residues in "
            f"{pdb_path.name}: {residue_list}{extra_msg}"
        )

    return full_sequence, chain_sequences


def extract_sequences_batch(protein_names: list, base_path: Path) -> dict:
    """Extract sequences from multiple PDB files."""
    sequences = {}
    errors = {}

    for protein_name in protein_names:
        pdb_path = base_path / protein_name / "protein.pdb"
        if not pdb_path.exists():
            errors[protein_name] = "PDB file not found"
            continue

        try:
            full_sequence, chain_sequences = extract_sequence_from_pdb(pdb_path)
            if not full_sequence:
                errors[protein_name] = "No valid sequence extracted"
            else:
                sequences[protein_name] = {
                    "full_sequence": full_sequence,
                    "chain_sequences": chain_sequences,
                }
        except Exception as e:
            errors[protein_name] = f"Error extracting sequence: {str(e)}"

    return sequences, errors


def _is_valid_embedding(embedding_path: Path, output_format: str) -> bool:
    """
    Check if an embedding file exists and is valid.

    Args:
        embedding_path: Path to the embedding file
        output_format: Format of the embedding file ('npz' or 'hdf5')

    Returns:
        True if the embedding file is valid, False otherwise
    """
    if not embedding_path.exists():
        return False

    try:
        if output_format == "npz":
            with np.load(embedding_path) as data:
                # Check for required keys
                required_keys = ['residue_embeddings', 'sequence_embedding', 'sequence']
                if not all(key in data.files for key in required_keys):
                    return False
                # Check embeddings have valid shape
                if data['residue_embeddings'].shape[0] == 0:
                    return False
        elif output_format == "hdf5":
            with h5py.File(embedding_path, 'r') as f:
                required_keys = ['residue_embeddings', 'sequence_embedding', 'sequence']
                if not all(key in f for key in required_keys):
                    return False
                if f['residue_embeddings'].shape[0] == 0:
                    return False
        return True
    except Exception:
        return False


def generate_esm_embeddings_batch(
    protein_names: list,
    base_path: Path,
    model_name: str = "esm2_t33_650M_UR50D",
    output_format: str = "hdf5",
    device: str = "auto",
) -> dict:
    """
    Generate ESM embeddings for a batch of proteins.

    Each chain in multi-chain proteins is processed separately through ESM,
    then the embeddings are concatenated in order to preserve chain boundaries.

    Returns:
        dict: Maps protein_name to tuple (success: bool, message: str)
    """
    results = {}
    model = None
    alphabet = None
    batch_converter = None

    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        elif device == "cpu":
            model = model.cpu()

        sequences, errors = extract_sequences_batch(protein_names, base_path)

        for protein_name, error in errors.items():
            results[protein_name] = (False, error)

        if not sequences:
            return results

        # Process each protein separately to handle multi-chain properly
        for protein_name, seq_data in sequences.items():
            try:
                full_sequence = seq_data["full_sequence"]
                chain_sequences = seq_data["chain_sequences"]

                # Process each chain separately
                chain_embeddings = []
                chain_ids = []
                chain_lengths = []

                for chain_id, chain_seq in chain_sequences:
                    # Process single chain
                    data = [(chain_id, chain_seq)]
                    _, _, chain_tokens = batch_converter(data)
                    chain_lens = (chain_tokens != alphabet.padding_idx).sum(1)

                    device_to_use = next(model.parameters()).device
                    chain_tokens = chain_tokens.to(device_to_use)

                    with torch.no_grad():
                        chain_results = model(
                            chain_tokens, repr_layers=[33], return_contacts=False
                        )

                    chain_repr = chain_results["representations"][33]

                    # Extract residue embeddings (skip BOS and EOS tokens)
                    start_idx = 1
                    end_idx = chain_lens[0] - 1

                    chain_emb = chain_repr[0, start_idx:end_idx].cpu().numpy()
                    chain_embeddings.append(chain_emb)
                    chain_ids.append(chain_id)
                    chain_lengths.append(len(chain_seq))

                    # Clean up immediately
                    del chain_results, chain_repr, chain_tokens

                # Concatenate all chain embeddings in order
                residue_embeddings = np.concatenate(chain_embeddings, axis=0)

                # Generate per-sequence representation via averaging
                sequence_embedding = residue_embeddings.mean(axis=0)

                # Save embeddings immediately
                output_path = base_path / protein_name / f"embeddings.{output_format}"

                if output_format == "hdf5":
                    with h5py.File(output_path, "w") as f:
                        f.create_dataset("residue_embeddings", data=residue_embeddings)
                        f.create_dataset("sequence_embedding", data=sequence_embedding)
                        f.create_dataset("sequence", data=full_sequence.encode("utf-8"))
                        f.attrs["model_name"] = model_name
                        f.attrs["sequence_length"] = len(full_sequence)
                        f.attrs["embedding_dim"] = residue_embeddings.shape[1]
                        f.attrs["num_chains"] = len(chain_sequences)

                        # Store chain information
                        chain_ids_encoded = [cid.encode("utf-8") for cid in chain_ids]
                        f.create_dataset("chain_ids", data=chain_ids_encoded)
                        f.create_dataset("chain_lengths", data=chain_lengths)

                elif output_format == "npz":
                    np.savez_compressed(
                        output_path,
                        residue_embeddings=residue_embeddings,
                        sequence_embedding=sequence_embedding,
                        sequence=full_sequence,
                        chain_ids=chain_ids,
                        chain_lengths=chain_lengths,
                        num_chains=len(chain_sequences),
                    )
                else:
                    results[protein_name] = (False, f"Unsupported output format {output_format}")
                    continue

                results[protein_name] = (True, f"seq_len={len(full_sequence)}, chains={len(chain_sequences)}")

                # Explicitly delete variables to free memory
                del residue_embeddings
                del sequence_embedding
                del chain_embeddings

            except Exception as e:
                results[protein_name] = (False, str(e))

        # Clear all large variables
        del sequences

        # Clear GPU memory after processing batch
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        # If model loading or batch processing fails, mark all as failed
        for protein_name in protein_names:
            results[protein_name] = (False, str(e))

    finally:
        # Ensure all variables are deleted
        if model is not None:
            del model
        if alphabet is not None:
            del alphabet
        if batch_converter is not None:
            del batch_converter

        import gc

        gc.collect()

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


@click.command()
@click.option(
    "--path",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Path to the directory containing protein subdirectories",
)
@click.option(
    "--model",
    "-m",
    default="esm2_t33_650M_UR50D",
    help="ESM model to use (default: esm2_t33_650M_UR50D)",
)
@click.option(
    "--output-format",
    "-o",
    default="npz",
    type=click.Choice(["hdf5", "npz"]),
    help="Output format for embeddings (default: hdf5)",
)
@click.option(
    "--batch-size",
    "-b",
    default=8,
    type=int,
    help="Number of proteins to process in each batch (default: 8)",
)
@click.option(
    "--n-jobs",
    "-j",
    default=1,
    type=int,
    help="Number of parallel jobs (default: 1, use 1 for GPU memory efficiency)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--monitor-memory",
    is_flag=True,
    help="Monitor GPU memory usage",
)
@click.option(
    "--device",
    "-d",
    default="cpu",
    type=click.Choice(["auto", "cpu", "cuda"]),
    help="Device to use for computation (default: auto)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force regeneration of embeddings even if they already exist",
)
def generate_embeddings(
    path: Path,
    model: str,
    output_format: str,
    batch_size: int,
    n_jobs: int,
    verbose: bool,
    monitor_memory: bool,
    device: str,
    force: bool,
) -> None:
    """
    Generate ESM embeddings for proteins in PDB format using batched processing.

    This tool processes protein PDB files and generates ESM embeddings for each
    protein. Each subdirectory should contain a file named 'protein.pdb'.

    Multi-chain proteins are handled correctly by processing each chain separately
    through ESM, then concatenating the embeddings in order. This preserves chain
    boundaries and ensures proper context for each chain.

    The tool uses batched processing for efficiency - multiple proteins are
    processed together in each batch, but embeddings are saved individually.

    The output includes:
    - Per-residue embeddings (token-level representations)
    - Per-sequence embeddings (averaged representations)
    - Original amino acid sequence
    - Chain IDs and lengths (for multi-chain proteins)

    Example usage:
        python generate_esm_embeddings.py --path data/proteins
        python generate_esm_embeddings.py -p data/proteins -m esm2_t12_35M_UR50D \\
            -o npz -b 16 -j 1 -v --monitor-memory
    """
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    click.echo(f"[INFO] Generating ESM embeddings using model: {model}")
    click.echo(f"[INFO] Input path: {path}")
    click.echo(f"[INFO] Output format: {output_format}")
    click.echo(f"[INFO] Batch size: {batch_size}")
    click.echo(f"[INFO] Device: {device}")

    # Show initial memory usage if monitoring
    if monitor_memory and device == "cuda" and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        click.echo(f"[INFO] Initial GPU Memory - Allocated: {memory_allocated:.2f}GB")

    try:
        protein_names = [d for d in os.listdir(path) if (path / d).is_dir()]
        if not protein_names:
            click.echo("[WARNING] No subdirectories found in the specified path!")
            return
    except Exception as e:
        click.echo(f"[ERROR] Error reading directory: {e}")
        return

    click.echo(f"[INFO] Found {len(protein_names)} protein directories")

    # Check for missing PDB files
    missing_files = []
    for protein_name in protein_names:
        pdb_file = path / protein_name / "protein.pdb"
        if not pdb_file.exists():
            missing_files.append(protein_name)

    # Filter out proteins without PDB files
    valid_proteins = [name for name in protein_names if name not in missing_files]

    if not valid_proteins:
        click.echo("[WARNING] No valid proteins found!")
        return

    # Check for existing embeddings (skip unless --force)
    skipped_proteins = []
    if not force:
        proteins_to_process = []
        for protein_name in valid_proteins:
            embedding_file = path / protein_name / f"embeddings.{output_format}"
            if embedding_file.exists() and _is_valid_embedding(embedding_file, output_format):
                skipped_proteins.append(protein_name)
            else:
                proteins_to_process.append(protein_name)
        valid_proteins = proteins_to_process

    if not valid_proteins:
        click.echo("[INFO] All proteins already have embeddings. Nothing to do.")
        return

    click.echo(f"[INFO] Processing {len(valid_proteins)} proteins")

    # Create batches of proteins
    batches = [
        valid_proteins[i : i + batch_size]
        for i in range(0, len(valid_proteins), batch_size)
    ]

    # Track results
    all_results = {}
    failed_proteins = []

    try:
        if n_jobs == 1:
            # Sequential batch processing
            for batch in tqdm(batches, desc="Processing batches"):
                batch_results = generate_esm_embeddings_batch(
                    batch, path, model, output_format, device
                )
                all_results.update(batch_results)

                # Force garbage collection after each batch
                import gc
                gc.collect()
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Monitor memory usage if requested
                if monitor_memory and device == "cuda" and torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    click.echo(f"[INFO] GPU Memory - Allocated: {memory_allocated:.2f}GB")

                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
        else:
            # Parallel processing with proper progress tracking
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                batch_results_list = Parallel(n_jobs=n_jobs, return_as="generator")(
                    delayed(generate_esm_embeddings_batch)(
                        batch, path, model, output_format, device
                    )
                    for batch in batches
                )
                for batch_results in batch_results_list:
                    all_results.update(batch_results)
                    pbar.update(1)

    except KeyboardInterrupt:
        click.echo("\n[WARNING] Embedding generation interrupted by user!")
        return
    except Exception as e:
        click.echo(f"[ERROR] Error during processing: {e}")
        return

    # Collect failed proteins
    for protein_name, (success, message) in all_results.items():
        if not success:
            failed_proteins.append((protein_name, message))
            if verbose:
                click.echo(f"[WARNING] {protein_name}: {message}")

    # Count results
    successful_count = sum(1 for _, (success, _) in all_results.items() if success)
    failed_count = len(failed_proteins)

    # Write failed extractions to log file
    if failed_proteins:
        log_file = path / "failed_embeddings.log"
        with open(log_file, 'w') as f:
            f.write("protein_name\terror\n")
            for protein_name, error in failed_proteins:
                f.write(f"{protein_name}\t{error}\n")

    # Also add missing PDB files to failures
    if missing_files:
        log_file = path / "failed_embeddings.log"
        mode = 'a' if failed_proteins else 'w'
        with open(log_file, mode) as f:
            if not failed_proteins:
                f.write("protein_name\terror\n")
            for protein_name in missing_files:
                f.write(f"{protein_name}\tPDB file not found\n")
                if verbose:
                    click.echo(f"[WARNING] {protein_name}: PDB file not found")

    # Print summary
    click.echo(f"""
=== ESM Embedding Generation Summary ===
Successful: {successful_count}
Skipped (existing): {len(skipped_proteins)}
Missing PDB files: {len(missing_files)}
Failed: {failed_count}
========================================
""")

    if (failed_proteins or missing_files) and not verbose:
        log_file = path / "failed_embeddings.log"
        click.echo(f"[INFO] {len(failed_proteins) + len(missing_files)} proteins failed. "
                   f"See {log_file} for details or run with -v for verbose output.")

    if failed_proteins or missing_files:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    generate_embeddings()

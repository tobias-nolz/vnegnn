# VN-EGNN Allosteric Site Prediction: Architecture and Data Flow Documentation

## Executive Summary

This codebase implements an evaluation framework for the **VN-EGNN** model, specifically adapted for **allosteric site
prediction** on the Allosteric Database (ASD) dataset. The project leverages PyTorch Lightning, Hydra for configuration
management, and provides a complete pipeline from data preparation to model evaluation.

---

## Table of Contents

1. [General Structure and Organization](#1-general-structure-and-organization)
2. [Data Files: Structure, Generation, and Usage](#2-data-files-structure-generation-and-usage)
    - 2.1 [Ligand Files](#21-ligand-files-ligandpdb)
    - 2.2 [Binding Information File](#22-binding-information-file-bindingnpz)
    - 2.3 [ESM Embeddings File](#23-esm-embeddings-file-embeddingsnpz)
    - 2.4 [Data Flow Summary](#24-data-flow-summary)
3. [ASD-Related Scripts Integration](#3-asd-related-scripts-integration)
4. [Ligand Extraction Process](#4-ligand-extraction-process-detailed-analysis)

---

## 1. General Structure and Organization

### 1.1 Project Layout

```
vnegnn/
├── configs/           # Hydra configuration files
├── data/              # Data storage (ASD datasets, processed data)
├── documentation/     # Project documentation
├── examples/          # Example PDB files
├── logs/              # Training and evaluation logs
├── notebooks/         # Jupyter notebooks for analysis
├── scripts/           # Data processing scripts
├── src/               # Core source code
│   ├── datasets/      # Dataset and DataModule implementations
│   ├── models/        # VN-EGNN and EquiPocket models
│   ├── modules/       # Metrics, losses, schedulers, callbacks
│   ├── utils/         # Utility functions
│   └── wrappers/      # Lightning wrappers for training
└── visualizations/    # Output visualizations
```

### 1.2 Configuration Management

The project uses **Hydra** for hierarchical configuration management:

| Config Directory | Purpose                                                       |
|------------------|---------------------------------------------------------------|
| `callbacks/`     | Training callbacks (early stopping, checkpointing, etc.)      |
| `data/`          | Dataset configurations (allosteric, equipocket, pdbbind2020)  |
| `experiment/`    | Experiment presets combining data, model, and trainer configs |
| `model/`         | Model architecture configurations                             |
| `trainer/`       | PyTorch Lightning trainer settings                            |
| `logger/`        | Logging configurations (Weights & Biases)                     |

### 1.3 Data Pipeline Overview

The data processing pipeline follows these stages:

```
1. Raw Data Acquisition
   └── Download PDB files from RCSB (prepare_pdb.py)
   └── Extract ligand structures from PDB files (prepare_ligand.py)

2. Feature Generation
   └── Generate ESM protein embeddings (generate_esm_embeddings.py)
   └── Extract binding information and residue depths (extract_binding_info.py)

3. Graph Construction
   └── Convert to PyTorch Geometric heterogeneous graphs (binding_dataset.py)
   └── Apply data augmentations (rotations, subsampling)

4. Training / Evaluation
   └── Train VN-EGNN model (train.py)
   └── Evaluate on test sets (eval.py)
```

### 1.4 Model Architecture

The **VN-EGNN** model is an E(n)-equivariant message passing neural network that:

- Takes protein graph representations with node features (ESM embeddings)
- Uses heterogeneous graph structure with atom nodes and global nodes
- Applies equivariant message passing layers that preserve E(3) symmetry
- Outputs binding site predictions with confidence scores

Key configuration parameters:

- **Input features**: 1301 (ESM embeddings + residue depth)
- **Hidden features**: 100
- **Layers**: 5
- **Dropout**: 0.1
- **Aggregation**: Mean aggregation for both nodes and coordinates

### 1.5 Evaluation Metrics

Two primary metrics are used:

| Metric                               | Description                                                              |
|--------------------------------------|--------------------------------------------------------------------------|
| **DCA (Distance to Closest Atom)**   | Measures if any predicted point is within threshold (4Å) of ligand atoms |
| **DCC (Distance to Closest Center)** | Measures if predictions are within threshold of binding site centers     |

For both, the `n_rank_dca_*` and `n_rank_dcc_*` metrics are evaluated, where `*` indicates different `n` values (from 0
to `num_global_nodes − 1`).

The value `n` indicates the number of additional top-ranked predictions considered beyond the number of true ligands for
the protein. Thus, `n_rank_dca_0` evaluates success using exactly `L` predictions, where `L` is the number of true
ligands, while `n_rank_dca_2` allows two additional predictions (i.e., `top-(L+2)` predictions).

---

## 2. Data Files: Structure, Generation, and Usage

This section provides detailed information about the key data files used in the pipeline, including how they are
generated and where they are used.

### 2.1 Ligand Files (`ligand_*.pdb`)

#### 2.1.1 File Structure

Ligand files are standard PDB files containing HETATM records for a single ligand molecule. Each file contains:

| Record Type | Description                                           |
|-------------|-------------------------------------------------------|
| `HETATM`    | Atom coordinates for ligand atoms (x, y, z positions) |
| `CONECT`    | Connectivity information between atoms (if present)   |
| `END`       | File terminator                                       |

**Example directory structure:**

```
data/allosteric-sites/allosteric/raw/1ABC/
├── protein.pdb        # Full protein structure
├── ligand_0.pdb       # First extracted ligand
├── ligand_1.pdb       # Second extracted ligand (if multiple)
├── binding.npz        # Processed binding information
└── embeddings.npz     # ESM embeddings
```

#### 2.1.2 Purpose in the Pipeline

Ligand files serve as **intermediate data** and are used:

1. **During `extract_binding_info.py` processing:**
    - Loaded via RDKit's `read_molecule()` function
    - Ligand atom coordinates are extracted to calculate binding site information
    - Used to determine which protein residues are within the binding threshold (default: 4Å)

2. **Information extracted from ligands:**
    - **Atom 3D coordinates** → Used to compute distances to protein atoms
    - **Binding residues** → Protein atoms within threshold distance of any ligand atom
    - **Binding site centers** → Mean position of binding residues for each ligand

**Ligands are NOT directly used during model training or inference** - their information is pre-processed into
`binding.npz`.

---

### 2.2 Binding Information File (`binding.npz`)

#### 2.2.1 File Contents

The `binding.npz` file is a NumPy compressed archive containing all binding-related data for a protein-ligand complex:

| Key                    | Type      | Shape                     | Description                                                                        |
|------------------------|-----------|---------------------------|------------------------------------------------------------------------------------|
| `binding_residues`     | `bool`    | `(num_residues,)`         | Binary mask: True if residue is part of binding site                               |
| `binding_site_centers` | `float64` | `(num_ligands, 3)`        | XYZ coordinates of each binding site center                                        |
| `res_coords`           | `float64` | `(num_residues, 3)`       | CA atom coordinates for each residue                                               |
| `res_names`            | `str`     | `(num_residues,)`         | Three-letter amino acid codes (e.g., "ALA", "GLY")                                 |
| `res_ids`              | `int`     | `(num_residues,)`         | PDB residue numbering                                                              |
| `chains`               | `str`     | `(num_residues,)`         | Chain identifiers (e.g., "A", "B")                                                 |
| `res_depths`           | `float64` | `(num_residues,)`         | Residue depth from protein surface (via MSMS, can be skipped for standard VN-EGNN) |
| `ligand_coords`        | `float64` | `(total_ligand_atoms, 3)` | All ligand atom coordinates concatenated                                           |
| `ligand_ids`           | `int`     | `(total_ligand_atoms,)`   | Ligand index for each atom (0, 1, 2, ...)                                          |

#### 2.2.2 Generation Process

Generated by `scripts/extract_binding_info.py`:

```python
# 1. Parse protein structure
protein = PDBParser().get_structure("protein", protein_path)
protein_info = extract_protein_info(protein)  # Extract all atom info

# 2. Load all ligand files
ligand_mols = [read_molecule(path) for path in ligand_paths]
ligand_infos = [extract_ligand_info(mol) for mol in ligand_mols]

# 3. Calculate binding sites
# For each protein atom, check if any ligand atom is within threshold
dists = np.linalg.norm(protein_coords[:, None] - ligand_coords[None], axis=-1)
binding_atoms = (dists <= threshold).any(axis=1)

# Binding site center = mean of binding atom coordinates
binding_site_center = binding_atoms_coords.mean(axis=0)

# 4. Filter to CA atoms only (one per residue)
ca_atoms = protein_info.atom_names == "CA"
binding_residues = binding_atoms[ca_atoms]
res_coords = protein_info.coords[ca_atoms]

# 5. Calculate residue depths (optional, requires MSMS)
depth_map = calculate_residue_depths(protein)

# 6. Save to binding.npz
np.savez(output_path, binding_residues=..., binding_site_centers=..., ...)
```

#### 2.2.3 Usage in Training/Evaluation

**During graph construction (`src/datasets/binding_dataset.py`):**

```python
def process_protein(path: Path):
    binding_info = np.load(path / "binding.npz")

    # Used for node features
    coords = binding_info["res_coords"]  # Atom positions
    res_names = binding_info["res_names"]  # One-hot encoding
    res_depths = binding_info["res_depths"]  # Additional feature

    # Used for labels/targets
    binding_residues = binding_info["binding_residues"]  # Binary classification target
    binding_sites = binding_info["binding_site_centers"]  # Regression target

    # Used for evaluation metrics
    ligand_coords = binding_info["ligand_coords"]  # DCA metric
    ligand_ids = binding_info["ligand_ids"]  # Separate ligands
```

**Role in the model:**

| Data                           | Usage                                                 |
|--------------------------------|-------------------------------------------------------|
| `res_coords`                   | Node positions in the graph (CA atoms)                |
| `res_names`                    | One-hot encoded as part of node features (21 classes) |
| `res_depths`                   | Appended to node features (1 dimension)               |
| `binding_residues`             | **Training target** - segmentation loss               |
| `binding_site_centers`         | **Training target** - position prediction loss        |
| `ligand_coords` + `ligand_ids` | **Evaluation** - DCA metric calculation               |

---

### 2.3 ESM Embeddings File (`embeddings.npz`)

#### 2.3.1 File Contents

The `embeddings.npz` file contains protein language model embeddings:

| Key                  | Type      | Shape                  | Description                    |
|----------------------|-----------|------------------------|--------------------------------|
| `residue_embeddings` | `float32` | `(num_residues, 1280)` | Per-residue ESM-2 embeddings   |
| `sequence_embedding` | `float32` | `(1280,)`              | Mean-pooled sequence embedding |
| `sequence`           | `str`     | -                      | Full amino acid sequence       |
| `chain_ids`          | `str[]`   | `(num_chains,)`        | Chain identifiers              |
| `chain_lengths`      | `int`     | `(num_chains,)`        | Number of residues per chain   |
| `num_chains`         | `int`     | -                      | Total number of chains         |

**Note:** The ESM-2 model used is `esm2_t33_650M_UR50D`, which produces 1280-dimensional embeddings.

#### 2.3.2 Generation Process

Generated by `scripts/generate_esm_embeddings.py`:

```python
# 1. Extract sequence from PDB (matching binding info extraction)
full_sequence, chain_sequences = extract_sequence_from_pdb(pdb_path)
# - Only standard residues (ATOM records, not HETATM)
# - Skip residues without CA atoms
# - Skip non-standard amino acids

# 2. Load ESM-2 model
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
batch_converter = alphabet.get_batch_converter()

# 3. Process each chain separately (for multi-chain proteins)
chain_embeddings = []
for chain_id, chain_seq in chain_sequences:
    _, _, chain_tokens = batch_converter([(chain_id, chain_seq)])
    with torch.no_grad():
        results = model(chain_tokens, repr_layers=[33])

    # Extract embeddings (skip BOS/EOS tokens)
    chain_emb = results["representations"][33][0, 1:-1]
    chain_embeddings.append(chain_emb)

# 4. Concatenate all chain embeddings
residue_embeddings = np.concatenate(chain_embeddings, axis=0)
sequence_embedding = residue_embeddings.mean(axis=0)

# 5. Save to embeddings.npz
np.savez_compressed(output_path, residue_embeddings=..., ...)
```

**Important:** The sequence extraction must exactly match `extract_binding_info.py` to ensure alignment between
embeddings and coordinates.

#### 2.3.3 Usage in Training/Evaluation

**During graph construction (`src/datasets/binding_dataset.py`):**

```python
def process_protein(path: Path):
    esm_features = np.load(path / "embeddings.npz")["residue_embeddings"]

    # Concatenate with one-hot residue encoding
    # Final node features: 21 (one-hot) + 1280 (ESM) = 1301 dimensions
    data["atom"].x = cat_features(res_to_one_hot(res_names), esm_features)

    # Global node features: mean-pooled ESM embeddings
    data["global_node"].x = esm_features.mean(axis=0)
```

**Model configuration (`configs/model/vnegnn.yaml`):**

```yaml
backbone:
  input_features: 1301  # 21 (one-hot) + 1280 (ESM-2)
```

---

### 2.4 Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA GENERATION                                │
└─────────────────────────────────────────────────────────────────────────────┘
     
     ┌──────────────┐    ┌─────────────────┐    
     │ ASD Dataset  │───►│ prepare_pdb.py  │    
     │   (.txt)     │    │ (download RCSB) │    
     └──────┬───────┘    └────────┬────────┘  
            │                     │
            │                     ▼
            │            ┌─────────────────┐
            │     ┌──────│   protein.pdb   │──────────────┐
            │     │      │ (full structure)│              │ 
            │     │      └────────┬────────┘              │
            │     │               │                       │
            │     │               ▼                       │
            │     │      ┌─────────────────┐              │
            └───────────►│prepare_ligand.py│              │
                  │      │ (extract HETATM)│              │
                  │      └────────┬────────┘              │
                  │               │                       │
                  │               ▼                       │
                  │      ┌─────────────────┐              │
                  │      │  ligand_*.pdb   │──────────────┤
                  │      │(individual ligs)│              │
                  │      └─────────────────┘              │
                  │                                       │
                  ▼                                       ▼
     ┌──────────────────────────┐            ┌──────────────────────────┐
     │generate_esm_embeddings.py│            │ extract_binding_info.py  │
     │  - Extract sequence      │            │  - Calculate distances   │
     │  - Run ESM-2 model       │            │  - Find binding sites    │
     │  - Per-residue embeds    │            │  - Compute depths (skip) │
     └────────────┬─────────────┘            └───────────┬──────────────┘
                  │                                      │
                  │                                      │
                  ▼                                      ▼
     ┌─────────────────────────┐             ┌────────────────────────┐
     │    embeddings.npz       │             │     binding.npz        │
     │  - residue_embeddings   │             │  - binding_residues    │
     │  - sequence_embedding   │             │  - binding_site_centers│
     └────────────┬────────────┘             │  - res_coords          │
                  │                          │  - ligand_coords       │
                  │                          └───────────┬────────────┘
                  │                                      │
                  │                                      │
                  │                                      │
                  └───────────────────┬──────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             GRAPH CONSTRUCTION                              │
│                            (binding_dataset.py)                             │
│                                                                             │
│  HeteroData Graph:                                                          │
│  ├── atom.pos       ← res_coords (CA positions)                             │
│  ├── atom.x         ← one_hot(res_names) + residue_embeddings + res_depths  │
│  ├── atom.y         ← binding_residues (training target)                    │
│  ├── atom.bindingsite_center ← binding_site_centers                         │
│  ├── global_node.x    ← mean(residue_embeddings)                            │
│  ├── global_node.pos  ← fibonacci_grid(centroid, radius)                    │
│  ├── ligand.ligand_coords  ← ligand_coords (for DCA metric)                 │
│  └── ligand.ligand_ids     ← ligand_ids                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING / EVALUATION                            │
│                                                                             │
│  Training:                                                                  │
│  ├── Segmentation loss: atom.y (binding_residues)                           │
│  ├── Position loss: atom.bindingsite_center                                 │
│  └── Confidence loss: predicted confidence scores                           │
│                                                                             │
│  Evaluation:                                                                │
│  ├── DCA: distance(predicted_pos, ligand_coords) ≤ threshold                │
│  └── DCC: distance(predicted_pos, binding_site_centers) ≤ threshold         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. ASD-Related Scripts Integration

### 3.1 ASD Dataset Overview

The Allosteric Database (ASD) contains:

- 2,422 allosteric proteins from 425 species
- 100,320 modulators (activators, inhibitors, regulators)
- Detailed annotations including binding site residues, chain information, and modulator details

The project uses the `ASD_Release_202309_AS.tar.gz` file (Release 5.1), which contains allosteric site descriptions.
For more info see [ASD Infos](ASD_information.md).

### 3.2 Script Overview

The ASD-specific scripts are located in `scripts/allosteric-sites/`:

| Script              | Purpose                                         |
|---------------------|-------------------------------------------------|
| `setup_data.py`     | Main orchestrator for ASD data preparation      |
| `prepare_pdb.py`    | Downloads PDB files from RCSB database          |
| `prepare_ligand.py` | Extracts ligand structures from downloaded PDBs |
| `__init__.py`       | Package initialization                          |

### 3.3 Integration Flow

```python
# Entry point: setup_data.py
setup_data(
    output_dir,  # Where to store processed data
    asd_dataset,  # DataFrame with ASD entries
    n_jobs,  # Parallel workers
    force_ligand_extraction,  # Re-extract even if exists
    clear_existing_pdb,  # Clear before downloading
    max_diff,  # Fuzzy matching tolerance
    only_lig, # Only consider "lig" modulators
    verbose  # Detailed logging
)
```

**Step 1: Input Validation**

- Validates required columns: `allosteric_pdb`, `modulator_chain`, `modulator_resi`
- Filters rows with missing values and logs them to `filtered_rows.log`

**Step 2: PDB Download (`prepare_pdb_directory`)**

- Normalizes PDB IDs (handles formats like "1ABC", "1ABC;2DEF", "1ABC_A")
- Downloads from `https://files.rcsb.org/download/{PDB_ID}.pdb`
- Creates directory structure: `{output_dir}/raw/{PDB_ID}/protein.pdb`
- Implements retry logic (3 attempts with backoff)
- Logs failed downloads to `failed_pdb_downloads.log`

**Step 3: Ligand Extraction (`prepare_ligands_from_asd`)**

- Parses chain and residue information from ASD format
- Extracts ligand atoms using BioPython
- Saves to `{PDB_ID}/ligand_{idx}.pdb`
- Implements fuzzy matching for residue IDs
- Logs failed extractions with diagnostics

### 3.4 Dataset Module Integration

The `AllostericDataModule` class extends `BindingDataModule`:

```python
class AllostericDataModule(BindingDataModule):
    @property
    def test_dataloader_indices(self) -> dict[str, int]:
        return {"allosteric": 0}

    def test_dataloader(self):
        return [self._create_dataloader("allosteric")]
```

This seamlessly integrates with the existing training/evaluation pipeline while maintaining the same graph construction
and data loading infrastructure.

---

## 4. Ligand Extraction Process (Detailed Analysis)

### 4.1 Input Format Handling

The ASD database specifies ligands in various formats, which `prepare_ligand.py` handles:

| Format                    | Example                          | Handling                         |
|---------------------------|----------------------------------|----------------------------------|
| Single ligand             | chain="A", residue="501"         | Direct extraction                |
| Multiple separate ligands | chain="A;B", residue="501;502"   | Semicolon split, extract each    |
| Same ligand across chains | chain="A,B", residue="501"       | Comma split, search both chains  |
| Multiple residues         | residue="401,402" or "1585/1586" | Extract as separate ligands      |
| Peptide ligands (range)   | residue="1-141"                  | **Skipped** (flagged as peptide) |

### 4.2 Parsing Functions

**`parse_chain_ids(chain_str)`**

```python
# Input: "A,B" or "A;B" or "A, B"
# Output: ["A", "B"]
```

**`parse_residue_id(residue_str)`**

```python
# Input: "501" → ([501], None)
# Input: "401,402" → ([401, 402], None)
# Input: "1-141" → ([], "Range format - peptide ligand")
```

### 4.3 Extraction Mechanism

The `LigandSelect` class filters atoms during PDB writing:

```python
class LigandSelect(Select):
    def accept_residue(self, residue):
        return (residue.get_parent().id in self.chain_ids and
                residue.id[1] == self.residue_id)
```

This selects only HETATM records matching the specified chain(s) and residue ID.

### 4.4 Fuzzy Matching

When exact matching fails and `max_diff > 0`:

```python
def find_closest_residue(pdb_file, chain_ids, target_residue, max_diff=2):
# Searches for HETATM residues within ±max_diff of target
# Returns closest match or None
```

This handles cases where:

- Residue numbering differs slightly between ASD database and PDB file
- PDB structures have been renumbered

### 4.5 Diagnostic System

When extraction fails, `diagnose_ligand_extraction()` provides detailed feedback:

| Issue                            | Diagnostic Example                                    |
|----------------------------------|-------------------------------------------------------|
| Chain not found                  | "Chain(s) ['X'] not found. Available: ['A', 'B']"     |
| Residue is standard (not ligand) | "Residue 501 exists but is not a HETATM (ligand)"     |
| Residue not found                | "Residue 501 not found. Closest HETATM: 502 (diff=1)" |
| No HETATM residues               | "No HETATM residues nearby. Closest residue: 450"     |

### 4.6 Validation

```python
def _is_valid_ligand_pdb(pdb_file: Path) -> bool:
# Checks for at least one ATOM or HETATM record
# Returns False for empty or malformed files
```

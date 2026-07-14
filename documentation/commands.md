# Commands used for Experiments

## VN-EGNN Training

Trains the base VN-EGNN model on sc-PDB (orthosteric sites only). This is both the model
whose orthosteric prior the zero-shot experiment below probes, and the reference point the
joint extension has to beat. See
[architecture_and_dataflow.md](architecture_and_dataflow.md) for the model design.

**Prerequisite:** the sc-PDB dataset is already downloaded and processed under
`data/data/sc-pdb`.

```bash
python src/train.py experiment=vnegnn
```

The run id printed to Weights & Biases is what you pass as `wandb_run_id` to every
evaluation below.

## Zero-Shot Allosteric Site Prediction

Measures how well the **orthosteric-only** VN-EGNN above transfers to **allosteric** sites
*without any allosteric training* — the zero-shot baseline the joint model is compared
against. The steps prepare the ASD dataset (allosteric only, no classifier), embed it, and
evaluate the existing checkpoint on it. Run them **in order**.

### 1. Data setup — download PDBs + extract the allosteric modulator

Downloads the ASD structures and extracts the annotated allosteric modulator ligand per
protein. Unlike the joint pipeline this uses **no** `--extract-orthosteric`: zero-shot has
no classification head, so only allosteric sites are needed.

- `--only-lig` keeps only small-molecule modulators (`modulator_class == 'Lig'`).
- `--exclude-ids` drops any ASD protein whose id appears in the sc-PDB train/valid splits,
  so the zero-shot test never overlaps the base model's training data.
- `--max-diff 0` requires an exact residue-id match between the ASD annotation and the
  structure (raise it to allow ±N fuzzy matching).

```bash
python scripts/allosteric-sites/setup_data.py \
  --asd-file data/ASD_dataset/ASD_Release_202309_AS.txt \
  --exclude-ids data/equipocket/sc-pdb/splits/train_ids_scpdb \
  --exclude-ids data/equipocket/sc-pdb/splits/valid_ids_scpdb \
  --max-diff 0 \
  --jobs 32 \
  --only-lig
```

### 2. Feature extraction — ESM embeddings + `binding.npz`

Generates `embeddings.npz` (ESM-2) and `binding.npz` for every protein under the ASD
`raw/` folder. `--data-dir` is omitted because it already defaults to
`data/allosteric-sites/allosteric`. `--skip-depth` skips the MSMS residue-depth
calculation, which is unused by VN-EGNN (and MSMS can hang on some structures).

```bash
python scripts/process_data.py \
  --jobs 32 \
  --device cuda \
  --skip-depth \
  --batch 128
```

### 3. Evaluation

Runs the orthosteric-trained checkpoint on the allosteric dataset (`+data=allosteric`) and
reports DCC/DCA rank-n — these numbers are the zero-shot baseline.

```bash
python src/eval.py \
  wandb_run_id=[wandb_run_id_from_train_run] \
  +data=allosteric
```

## Joint Orthosteric + Allosteric Training

This trains VN-EGNN jointly on sc-PDB (orthosteric) and the ASD training split
(allosteric), and additionally learns to classify each predicted site as
orthosteric vs allosteric. See
[architecture_and_dataflow.md](architecture_and_dataflow.md) and
[allosteric_extension.md](allosteric_extension.md) for the design.

**Prerequisite:** the sc-PDB (orthosteric) dataset is already downloaded and processed
under `data/data/sc-pdb`. The steps below prepare the ASD (allosteric) dataset, build a
leakage-free split, train jointly, and evaluate. Run them **in order**.

### 1. Data setup — download PDBs + extract ligands (with orthosteric augmentation)

Downloads the ASD structures and extracts the annotated **allosteric** modulator per
protein. `--extract-orthosteric` additionally harvests the other co-crystallized
drug-like ligands as **orthosteric** (class 0) sites, so a protein carries both classes
and the classifier learns pocket geometry rather than dataset identity. This is a
heuristic (see [allosteric_extension.md](allosteric_extension.md) and
`scripts/allosteric-sites/prepare_orthosteric.py`) — a non-modulator ligand is not
guaranteed to be orthosteric. Omit `--extract-orthosteric` to get an allosteric-only
dataset (the classification head is then not meaningful).

```bash
python scripts/allosteric-sites/setup_data.py \
  --asd-file data/ASD_dataset/ASD_Release_202309_AS.txt \
  --exclude-ids data/data/sc-pdb/splits/train_ids_scpdb \
  --exclude-ids data/data/sc-pdb/splits/valid_ids_scpdb \
  --only-lig --jobs 32 \
  --extract-orthosteric \
  --ortho-min-heavy-atoms 6
```

### 2. Feature extraction — ESM embeddings + `binding.npz`

Generates `embeddings.npz` (ESM-2) and `binding.npz` for every protein under the ASD
`raw/` folder. Proteins that received an orthosteric ligand in step 1 get a per-site
`site_types` array in their `binding.npz` (0 = orthosteric, 1 = allosteric).
`--skip-depth` skips the MSMS residue-depth calculation, which is unused by VN-EGNN.

```bash
python scripts/process_data.py \
  --data-dir data/allosteric-sites/allosteric \
  --jobs 32 \
  --device cuda \
  --batch 128 \
  --skip-depth
```

### 3. Sequence-identity split of the ASD dataset

Cluster all ASD sequences (and the sc-PDB train/valid sequences, to remove
cross-dataset leakage) with mmseqs2 at 30% identity and assign whole clusters to
train/valid/test. Requires [`mmseqs`](https://github.com/soedinglab/MMseqs2) on PATH.

```bash
python scripts/allosteric-sites/make_splits.py \
  --asd-dir data/allosteric-sites/allosteric \
  --scpdb-dir data/data/sc-pdb \
  --min-seq-id 0.3 \
  --coverage 0.8 \
  --ratios 0.5 0.25 0.25 \
  --suffix mmseqs30 \
  --jobs 32
```

This writes `train_ids_allosteric_mmseqs30`, `valid_ids_allosteric_mmseqs30`, and
`test_ids_allosteric_mmseqs30` under `data/allosteric-sites/allosteric/splits/`.
ASD proteins homologous to any sc-PDB train/valid protein are routed to the ASD
*train* fold so they can never leak into valid/test.

### 4. Joint training

```bash
python src/train.py experiment=vnegnn_joint
```

The `split_suffix` (default `mmseqs30`) is set in `configs/data/joint.yaml` and must
match the suffix used in step 3. Training draws orthosteric/allosteric samples ~50/50
(`balance_classes: true`) so the larger sc-PDB set does not dominate; set it to `false`
to train on the natural mixture.

In Weights & Biases watch:

- `train/class_loss` — the classification objective.
- `val/dcc_ranked_allo` / `val/dcc_ranked_ortho` — ranked DCC split by site type. The
  checkpoint is selected on `val/dcc_ranked_allo` (allosteric localization is the goal);
  change `callbacks.model_checkpoint.monitor` in `configs/experiment/vnegnn_joint.yaml` to
  `val/dcc_ranked` (pooled) or `val/dcc_ranked_ortho` to weight orthosteric retention.
- `val/class_auroc_detected` — classification quality on *found* pockets only. This is
  only meaningful once ASD proteins carry both classes (step 1 with
  `--extract-orthosteric`); without it the classifier target collapses to dataset
  identity and the AUROC is trivially ~1.0.

### 5. Evaluation

**Full evaluation (localization on every test set + orthosteric-vs-allosteric
classification).** Use the joint datamodule so the test set is COACH420 / HOLO4K /
PDBbind2020 (orthosteric) + the ASD test split (allosteric). The detection-conditioned
classifier metric (`classifier/auroc_detected`) pools class-0 detections from the
orthosteric benchmarks with class-1 detections from the allosteric set, so it needs both:

```bash
python src/eval.py wandb_run_id=[RUN_ID] +data=joint
```

**Allosteric-only comparison to the zero-shot baseline.** Evaluates just the ASD test
split; the prediction CSV additionally carries an `allosteric_prob_0` column per predicted
site. (Here every ground-truth site is allosteric, so `classifier/auroc_detected` is
skipped — one class only.)

```bash
python src/eval.py wandb_run_id=[RUN_ID] +data=allosteric
```

### 6. P2Rank baseline

Run [P2Rank](https://github.com/rdk/p2rank) (`prank` on PATH) on a split and report
DCC/DCA rank-n with the same success definition as the VN-EGNN evaluator:

```bash
python scripts/allosteric-sites/compare_p2rank.py \
  --asd-dir data/allosteric-sites/allosteric \
  --split-file data/allosteric-sites/allosteric/splits/test_ids_allosteric_mmseqs30 \
  --threshold 4.0 \
  --num-ranks 8 \
  --threads 8
```
# Allosteric-Site Extension

This document describes the changes that extend VN-EGNN from a generic binding-site
predictor (trained on sc-PDB only) into a model that is **jointly trained on
orthosteric (sc-PDB) and allosteric (ASD) sites** and additionally **classifies each
predicted site as orthosteric or allosteric**.

The motivation comes from the zero-shot study: the sc-PDB-only model detects only ~11% of allosteric sites at rank-0
because it has internalized a geometric prior toward deep orthosteric pockets. The extension addresses this with (a)
joint training and (b) an explicit
allosteric-classification head.

**Measured outcomes live in [`results/asd_results.md`](results/asd_results.md)** — every run, the
P2Rank baseline, both ablations and the evaluation denominators, taken from the W&B run summaries.
This document covers the design; that one covers what it produced.

## Summary of changes

### Model — a second virtual-node head

`src/models/vnegnn/vnegnn.py`

- Added `classifier_mlp`, a per-virtual-node MLP mirroring the existing
  `confidence_mlp`, that outputs a single logit = allosteric score of the site that
  virtual node represents.
- `VNEGNN.forward` now returns **5** tensors:
  `(x_atom, pos_global_node, x_global_node, confidence_out, class_out)`.
- The confidence module is unchanged and still only responsible for *ranking*; the new
  head is a separate *classifier*. (This is the "1 confidence module + 1 classifier"
  option from the project notes, chosen over two confidence modules.)

### Loss — a joint classification objective

`src/wrappers/bindingsites.py` (`BindingSitesLoss`)

- New `classification_loss` (default `BCEWithLogitsLoss`) and
  `classification_loss_weight`.
- **Per-site target**: each virtual node is labelled with the site type of its
  *nearest annotated site center*, reusing the `confidence_assign_index` knn already
  computed for the confidence term. (This replaced the initial per-protein broadcast —
  see the discussion at the end of this document.)
- Optional `classification_foreground_dist`: when set, only virtual nodes within that
  distance (Angstrom) of a real site contribute to the classification loss; background
  nodes far from any site are ignored. `null` supervises every virtual node.
- Total loss = `pos + seg + confidence + classification` (each weighted).

### Labels — `site_type` through the data layer

`src/datasets/utils.py`, `src/datasets/binding_dataset.py`

- `create_hetero_graph(..., site_type=0)` stores a graph-level scalar
  `data.site_type` (collates to `[num_graphs]`) plus a per-center
  `data["atom"].bindingsite_site_type`.
- `BindingDataset` takes `site_type` and includes it in the processed-cache hash so
  orthosteric and allosteric caches never collide.
- Convention: **sc-PDB and the orthosteric benchmarks = 0, ASD = 1**.

### Data — joint train/val/test datamodule

`src/datasets/joint_dataset.py` (`JointBindingDataModule`), `configs/data/joint.yaml`

- Train/val batches concatenate sc-PDB (`site_type=0`) and the ASD training split
  (`site_type=1`) so every mini-batch contains both classes.
- **Class-balanced sampling** (`balance_classes: true`, default): sc-PDB has ~orders of
  magnitude more proteins than ASD, so a plain shuffle would make almost every batch
  orthosteric — starving both allosteric localization and the classifier. The training
  loader instead uses a `WeightedRandomSampler` with inverse-frequency weights, so the two
  classes are drawn ~50/50. Validation is left at the natural (unbalanced) distribution.
  Set `balance_classes: false` to train on the raw mixture.
- Test set = the established VN-EGNN orthosteric benchmarks (COACH420, HOLO4K,
  PDBbind2020) + the held-out ASD allosteric split.
- `train_valid_split` falls back to `scpdb` via `${oc.select:...}` so `+data=joint` also
  composes under `eval.py` (which has no global `train_valid_split`).

### Splits — sequence-identity based, leakage-free

`scripts/allosteric-sites/make_splits.py` — **published ID lists and provenance in
[`splits.md`](splits.md)**; the split actually used by every run is the `mmseqs30` suffix.

- mmseqs2 clustering at 30% identity; whole clusters go to one split.
- ASD proteins sharing a cluster with any sc-PDB train/valid protein are forced into
  the ASD train fold, so test/valid folds are non-homologous to *all* training data.

### Prediction / metrics

- `src/utils/misc.py` `multi_predictions` returns the class logits too.
- `predict_step` writes `allosteric_prob` (sigmoid of the class logit) into the
  prediction CSVs (`allosteric_prob_0`).
- Validation logs `val/class_acc` / `val/class_auroc` (and the detection-conditioned
  `val/class_*_detected`, see below).
- **Per-class localization**: `val/dcc_ranked_ortho` and `val/dcc_ranked_allo` split the
  ranked DCC by site type. The joint val set is orthosteric-dominated, so the pooled
  `val/dcc_ranked` mostly tracks sc-PDB; the per-class metrics expose allosteric
  localization on its own. They are only logged in steps where that class is present, so a
  single-class run never divides by zero.
- **Checkpoint selection**: `configs/experiment/vnegnn_joint.yaml` overrides
  `callbacks.model_checkpoint.monitor` to `val/dcc_ranked_allo` (mode `max`), since
  reliable allosteric localization is the goal. Switch it to `val/dcc_ranked` (pooled) or
  `val/dcc_ranked_ortho` to weight orthosteric retention.
- `strict_loading = False` on `BindingSitesWrapper` so checkpoints trained before the
  classifier head (e.g. `best_zero_shot.ckpt.ckpt`) still load and evaluate.
- **Eval robustness**: `evaluate_protein_predictions` asserts that a protein's per-site
  `site_types` aligns with its centers/ligand ids before filtering, so a malformed
  extraction fails loudly instead of silently miscounting the allosteric benchmark.

### Config / experiment

- `configs/model/vnegnn.yaml`: `classification_loss` + `classification_loss_weight` +
  `classification_foreground_dist`.
- `configs/data/joint.yaml`: `balance_classes` (class-balanced sampling) and the
  `train_valid_split` fallback for eval.
- `configs/experiment/vnegnn_joint.yaml`: entry point
  (`python src/train.py experiment=vnegnn_joint`); also sets the checkpoint monitor to
  `val/dcc_ranked_allo`.
- Full evaluation (all test sets + orthosteric-vs-allosteric classifier AUROC):
  `python src/eval.py wandb_run_id=[RUN] +data=joint`. `+data=allosteric` remains the
  allosteric-only comparison to the zero-shot baseline.

### Baseline comparison

`scripts/allosteric-sites/compare_p2rank.py` — runs P2Rank and reports DCC/DCA rank-n
with the same success definition as `evaluate_protein_predictions`.

## Files touched

| File                                              | Change                                                                 |
|---------------------------------------------------|------------------------------------------------------------------------|
| `src/models/vnegnn/vnegnn.py`                     | new `classifier_mlp`, 5-tuple output                                   |
| `src/wrappers/bindingsites.py`                    | classification loss + metrics, `predict_step`, `strict_loading`        |
| `src/datasets/utils.py`                           | `site_type` on the graph                                               |
| `src/datasets/binding_dataset.py`                 | `site_type` param + cache hash                                         |
| `src/utils/misc.py`                               | `multi_predictions` returns class logits                               |
| `src/datasets/joint_dataset.py`                   | **new** joint datamodule + class-balanced `WeightedRandomSampler`      |
| `configs/data/joint.yaml`                         | **new**; `balance_classes` + eval-safe `train_valid_split` fallback    |
| `configs/experiment/vnegnn_joint.yaml`            | **new**; checkpoint monitor = `val/dcc_ranked_allo`                    |
| `configs/model/vnegnn.yaml`                       | classification loss config                                             |
| `scripts/allosteric-sites/make_splits.py`         | **new** mmseqs split                                                   |
| `scripts/allosteric-sites/compare_p2rank.py`      | **new** P2Rank baseline                                                |
| `scripts/allosteric-sites/prepare_orthosteric.py` | **new** orthosteric-candidate ligand extraction                        |
| `scripts/allosteric-sites/setup_data.py`          | `--extract-orthosteric` / `--ortho-min-heavy-atoms` flags              |
| `scripts/extract_binding_info.py`                 | writes per-site `site_types` in `binding.npz`                          |
| `src/utils/misc.py`                               | `site_type_filter` in evaluators; `collect_site_classifications`       |
| `src/eval.py`                                     | allosteric eval filter; pooled detection-conditioned classifier metric |
| `src/wrappers/bindingsites.py`                    | `val/class_*_detected` + per-class `val/dcc_ranked_{ortho,allo}`       |
| `src/utils/misc.py`                               | `site_types` alignment guard in `evaluate_protein_predictions`         |

## Labeling: per-site

Each virtual node is labelled with the site type of its **nearest annotated site
center**, using the `confidence_assign_index` knn already computed in
`BindingSitesLoss` and the per-center `data["atom"].bindingsite_site_type` produced in
`create_hetero_graph`. The validation metric uses the same nearest-center scheme. An
optional `classification_foreground_dist` restricts the loss/metric to virtual nodes
within a distance of a real site, so background nodes are not forced into a class. This
makes the target **local** rather than protein-global: on a protein that contains sites
of different classes, each virtual node is supervised by the pocket it is actually near.

### Where the per-center labels come from

- **sc-PDB / ASD without extra sites**: single-class. `binding.npz` has no `site_types`,
  and the datamodule's per-dataset `site_type` (0 for sc-PDB, 1 for ASD) fills every
  center — behaviour is unchanged.
- **ASD with orthosteric augmentation** (`setup_data.py --extract-orthosteric`):
  `scripts/allosteric-sites/prepare_orthosteric.py` harvests non-modulator, drug-like
  co-crystallized HETATMs as orthosteric (0) sites, saved as `ligand_ortho_*.pdb`.
  `scripts/extract_binding_info.py` then writes a per-center `site_types` array into
  `binding.npz` (any `ligand_ortho_*` site = 0, the annotated modulator site = 1), which
  `create_hetero_graph` uses to populate `bindingsite_site_type`. Now a single protein
  carries both classes, so the classifier must use pocket geometry, not dataset identity.

**Caveat on the orthosteric heuristic.** "A non-modulator co-crystallized ligand" is only
an approximation of "orthosteric": it can be a cofactor, a second allosteric site, or a
crystallization additive. We mitigate with a HETATM blocklist (water/ions/buffers/cryo)
and a minimum heavy-atom count, but the labels are noisy. A principled active-site source
(UniProt `ACT_SITE`/`BINDING` or M-CSA, mapped to PDB numbering via SIFTS) would be more
accurate and is the recommended next upgrade.

**Keeping the allosteric benchmark clean.** Because orthosteric augmentation adds
orthosteric ground-truth sites to ASD proteins, the allosteric DCC/DCA evaluation would
otherwise be contaminated. `evaluate_protein_predictions` / `evaluate_all_proteins` take a
`site_type_filter`, and `src/eval.py` passes `1` for the `allosteric` dataloader, so the
reported allosteric numbers still count only allosteric sites — comparable to the
zero-shot baseline.

### Decoupling detection from classification at eval

A raw classifier metric over *all* virtual nodes is dominated by background nodes that
never landed on a pocket, so it does not answer the question we care about: **given a
pocket the model actually found, is its type predicted correctly?** Both the validation
and evaluation paths now report a *detection-conditioned* metric.

- **Validation** (`src/wrappers/bindingsites.py`): `val/class_acc_detected` and
  `val/class_auroc_detected` are computed only over virtual nodes within the DCC
  threshold (`hparams.threshold`, 4 Å) of a true site. (`val/class_acc` /
  `val/class_auroc` remain, computed over all — or foreground-masked — nodes.)
- **Evaluation** (`src/utils/misc.py` `collect_site_classifications` +
  `src/eval.py`): for every ground-truth site, the nearest predicted virtual node is
  found; if it is within 4 Å (a successful detection) a `(true_class, allosteric_prob)`
  pair is recorded. These are pooled across **all** test datasets — the orthosteric
  benchmarks contribute class-0 detections, the allosteric set class-1 — and reported as
  `classifier/acc_detected`, `classifier/auroc_detected`, and `classifier/n_detected`.
  Pooling both classes is what makes the AUROC a genuine binary-discrimination number.

`classification_foreground_dist` is a coarse training-time analogue of the same idea
(restricting the *loss* to near-site nodes).

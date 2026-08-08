# ASD (allosteric) evaluation results

Canonical record of every allosteric evaluation behind Section 5 of the thesis. All values are
copied from the W&B run summaries under `logs/eval/runs/<dir>/wandb/*/files/wandb-summary.json`,
rounded to three decimals; nothing here is re-derived by hand.

## How to read the numbers

- **DCC_n** — fraction of ground-truth site *centres* with a predicted virtual node within 4 Å,
  allowing the top `num_sites + n` ranked predictions. **DCA_n** — same, but distance to any
  *ligand atom* (looser, so DCA > DCC throughout).
- **Aggregation is a macro-average over proteins, not over centres.** `compute_metric_ratios`
  (`src/utils/misc.py`) computes `round(hits / num_ligs, 2)` per protein, and `src/eval.py`
  averages those per-protein ratios. A protein with one site counts as much as one with four, and
  a pooled per-centre rate recomputed from these tables will not reproduce them exactly.
- **DCC_0** = the ranking head must put the right node first. **DCC_7** = the ranking-free
  localisation ceiling (Mean-Shift collapses the extra nodes, so the curve is saturated by rank 7;
  verified at K=16 and K=32, where rank-15 / rank-31 equal rank-7).
- **auroc_detected** — allosteric-vs-orthosteric classifier AUROC restricted to sites the model
  actually localised, pooled over the orthosteric benchmarks (class 0) and the ASD test set
  (class 1).

## Denominators

| eval mode | proteins in prediction file | proteins scored | allosteric centres |
|---|---:|---:|---:|
| single chain (matched) | 168 | **150** | **158** |
| full assembly | 167 | **149** | **156** |

Proteins are skipped when no site of the requested class survives the `max_center_dist = 8 Å`
and `site_type == 1` filters. The two blocks use different denominators and are **not**
comparable across the mid-rule.

Two facts about the ground truth that the denominator hides:

- 50 of the 168 evaluated ASD test proteins have **no `site_types` array** in `binding.npz`. For
  these the `site_type_filter` is skipped entirely, so all their sites (51 of the 158) count as
  allosteric. The assumption is that a protein without harvested orthosteric ligands is
  allosteric-only, which holds here (≈1 site each), but it covers a third of the benchmark.
- 18 of the 168 carry `site_types` but **no site labelled allosteric**, and are dropped. For ASD
  entries, which are defined by an annotated modulator, that is a labelling gap worth knowing
  about. It is the test-set counterpart of the 32 orthosteric-only ASD training proteins.

## All runs

Allosteric test split (30 % MMseqs2 identity), plus orthosteric rank-0 DCC for context.

| run | train id | eval dir | aDCC_0 | aDCC_7 | aDCA_0 | aDCA_7 | c420 | holo4k | pdbbind | AUROC | acc | n_det |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot, full-asm | `a5om5ixz` | `2026-07-28_14-12-45` | 0.054 | 0.107 | 0.087 | 0.181 | 0.646 | 0.626 | 0.646 | – | – | – |
| Zero-shot, matched | `a5om5ixz` | `2026-07-28_14-08-10` | 0.140 | 0.167 | 0.187 | 0.207 | 0.646 | 0.626 | 0.646 | – | – | – |
| Base | `ud3y8amg` | `2026-07-23_16-50-13` | 0.067 | 0.114 | 0.134 | 0.282 | 0.465 | 0.526 | 0.477 | 0.699 | 0.948 | 7651 |
| SC, full-asm eval | `xf7l2qm7` | `2026-07-23_23-34-03` | 0.087 | 0.168 | 0.156 | 0.263 | 0.526 | 0.567 | 0.565 | 0.729 | 0.850 | 7762 |
| SC | `xf7l2qm7` | `2026-07-24_09-06-22` | 0.140 | 0.200 | 0.222 | 0.315 | 0.526 | 0.567 | 0.565 | 0.829 | 0.851 | 7767 |
| K16 | `rpx2fuzn` | `2026-07-24_15-14-52` | 0.133 | 0.220 | 0.188 | 0.388 | 0.546 | 0.569 | 0.562 | 0.658 | 0.910 | 8099 |
| loMSE | `rbfqpluz` | `2026-07-25_07-30-16` | 0.155 | 0.202 | 0.235 | 0.355 | 0.492 | 0.565 | 0.509 | 0.638 | 0.917 | 8301 |
| **sphere** | `9ocz3qmi` | `2026-07-25_13-54-12` | **0.170** | 0.223 | 0.243 | 0.373 | 0.547 | 0.558 | 0.551 | 0.698 | 0.913 | 8046 |
| K32 | `44jnpld6` | `2026-07-25_19-19-18` | 0.150 | **0.240** | 0.225 | **0.435** | 0.542 | 0.544 | 0.522 | **0.834** | 0.909 | 8572 |
| reg | `rmbs3zry` | `2026-07-26_14-19-08` | 0.168 | 0.222 | 0.222 | 0.308 | 0.503 | 0.547 | 0.561 | 0.712 | 0.911 | 8155 |
| car | `do5iryky` | `2026-07-27_15-45-19` | 0.088 | 0.122 | 0.162 | 0.288 | 0.493 | 0.481 | 0.464 | 0.600 | 0.916 | 8203 |

The zero-shot rows are the original sc-PDB-only checkpoint loaded with `strict_loading=False`; it
has no site-type head, so its classifier columns are meaningless and are omitted. Its orthosteric
columns are the published re-training reproduction, not a joint run.

**K32 + class-aware re-rank (the deployment deliverable).** A re-ranking of the frozen K32
predictions by `z(confidence) + z(allosteric_prob)`, produced with `rerank_probe.py`; no retraining.

| ranking | DCC_0 | DCC_2 | DCC_7 | DCA_0 | DCA_7 |
|---|---:|---:|---:|---:|---:|
| confidence (K32) | 0.150 | 0.227 | 0.240 | 0.225 | 0.435 |
| **conf + allosteric prob** | 0.133 | **0.240** | 0.240 | **0.268** | 0.435 |

It has the best DCA_0 of any configuration and reaches the DCC ceiling by rank 2 instead of rank 4,
at the cost of DCC_0. Applied to the K=16 `sphere` model, whose classifier is weaker (0.698 against
0.834), the same re-ranking *lowers* DCA_0 to 0.203, so the gain is conditional on the classifier.

## P2Rank baseline

`scripts/allosteric-sites/compare_p2rank.py --matched`, run on the same single-chain structures and
scored on the identical denominator (150 proteins / 158 centres). Per-protein hits are in
`data/allosteric-sites/p2rank_matched_metrics.csv`.

| rank | DCC | DCA |
|---:|---:|---:|
| 0 | 0.217 | 0.383 |
| 1 | 0.310 | 0.497 |
| 2 | 0.347 | 0.587 |
| 3 | 0.410 | 0.643 |
| 7 | 0.433 | 0.673 |

P2Rank outperforms every VN-EGNN configuration on this split, at rank-0 and by a wide margin at the
ceiling. What VN-EGNN adds instead is per-pocket orthosteric/allosteric typing, which P2Rank does
not do.

## Label-scheme ablation

Two runs at the `sphere` config, each flipping one label toggle. The per-site-with-augmentation
control is the `sphere` row above (AUROC 0.698).

| config | AUROC | acc | n_det | aDCC_0 | aDCC_7 | aDCA_0 | aDCA_7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| per-protein labels (`s91233ey`) | 0.654 | 0.785 | 7832 | 0.127 | 0.147 | 0.182 | 0.262 |
| no ortho augmentation (`t5g7p7vp`) | 0.718 | 0.851 | 7279 | 0.068 | 0.168 | 0.228 | 0.362 |

Per-protein labelling is the worst of the three, which vindicates per-site labelling. Removing the
orthosteric augmentation *raises* the pooled AUROC, but that is a metric artefact: this metric pools
allosteric-test positives against orthosteric-benchmark negatives, so it rewards clean cross-dataset
separation, and with no within-protein orthosteric pockets to type the classifier reduces to exactly
that. The augmentation's actual purpose, typing two different-class pockets inside one protein, is
untestable by a metric that filters each dataset to a single class.

## Positive-class-weight ablation (400 epochs)

Matched pair differing only in `w_p`, with `classification_loss_weight` pinned at 1.0 in both so the
magnitude inflation is uncompensated and visible. Both otherwise use the `sphere` config.

| config | c420 DCC_0 | holo4k | pdbbind | aDCC_0 | aDCA_0 | AUROC | acc | n_det |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: `w_p=1`, `λ=1.0` (`yr32vrnu`) | 0.549 | 0.562 | 0.578 | 0.140 | 0.268 | 0.719 | 0.900 | 8125 |
| B: `w_p=3.2`, `λ=1.0` (`j8kkvp6o`) | 0.503 | 0.527 | 0.507 | 0.127 | 0.223 | 0.811 | 0.889 | 7918 |
| C: `w_p=3.2`, `λ=0.5` (= `sphere`) | 0.547 | 0.558 | 0.551 | 0.170 | 0.243 | 0.698 | 0.913 | 8046 |

Findings:

- **Dissociation, but not a clean one.** Rank-0 falls on all eight dataset/metric pairs (mean
  −9.5 %, worst −16.8 % on allosteric DCA); rank-7 moves only −1.9 % and rises in two of eight.
  The ~5:1 ratio is the effect the experiment was built to detect, but rank-7 is not flat, so `w_p`
  costs a little localisation as well as ranking.
- **Ranking damage is the clearest signal.** Validation confidence variance falls 0.207 → 0.183.
  Random-ranked DCC *improves* 0.278 → 0.303 while confidence-ranked improves only 0.327 → 0.338,
  so the module's advantage over chance shrinks from +17.8 % to +11.6 %.
- **Localisation drifts slightly:** validation position loss 2.505 → 2.670, mean prediction
  distance 5.80 → 6.20 Å (~7 %). This is the part of the rank-7 change the ranking argument does
  not explain.
- **Magnitude factor 3.17×**, measured on validation classification loss (1.164 → 3.695). It cannot
  be read from training loss: **both arms drive `train/class_loss` to 0.000** by the end of the
  schedule, i.e. the classifier fully fits the training set.
- **λ_cls = 0.5 is a trade, not a fix.** It recovers 94 % / 91 % / 63 % of arm B's rank-0 DCC loss
  on COACH420 / HOLO4K / PDBbind and overshoots on the allosteric set (0.170 against the control's
  0.140), but returns the classifier to 0.698 AUROC, *below* the `w_p = 1` control. A weight between
  0.5 and 1 might hold both; this two-point sweep cannot resolve it.

Caveat on row C: it is the `sphere` run, trained 2026-07-25 with the same command and defaults, not
a purpose-built third arm trained alongside A and B on 2026-08-07. Checkpoint selection also differs
(epochs 254 / 299 / 389), all by the same early-stopping rule on validation DCC — which optimises
DCC, not AUROC, so the localisation comparison is more trustworthy than the AUROC comparison.

## Reproduction

```bash
# zero-shot, both denominators (no retraining; strict_loading=False skips the absent classifier head)
python src/eval.py wandb_run_id=a5om5ixz +data=joint data.graph_info.number_of_global_nodes=8 \
    data.single_chain_allosteric=true        # matched
python src/eval.py wandb_run_id=a5om5ixz +data=joint data.graph_info.number_of_global_nodes=8

# joint runs (K=16 + ranking head are the current defaults; K=8 runs need the override)
scripts/allosteric-sites/train_resumable.sh data.single_chain_allosteric=true
python src/eval.py wandb_run_id=<id> +data=joint data.single_chain_allosteric=true

# class-aware deployment re-rank of a finished evaluation
python scripts/allosteric-sites/rerank_probe.py \
    logs/eval/runs/<dir>/predictions_allosteric.csv --label deploy   # read conf_z_plus_allo_z
# K is inferred from the file; --strategy conf_z_plus_allo_z limits it to the deployment row

# P2Rank baseline on the matched denominator
python scripts/allosteric-sites/compare_p2rank.py \
    --asd-dir data/allosteric-sites/allosteric \
    --split-file data/allosteric-sites/allosteric/splits/test_ids_allosteric_mmseqs30 \
    --matched --proteins-csv logs/eval/runs/2026-07-28_14-08-10-320556/predictions_allosteric.csv
```

`K` is part of the processed-cache key, so changing it rebuilds the caches; run
`prebuild_caches.py` first.

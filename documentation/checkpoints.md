# Released checkpoints

The two models the thesis reports: the best rank-0 localiser (`sphere`) and the deployment
model (`K32` + class-aware re-ranking). The weights are published as **release assets** under
the [`v1.0-thesis`](https://github.com/tobias-nolz/vnegnn/releases/tag/v1.0-thesis) tag, not
committed to the repository, so a clone stays small.

| file                                                                                                                                         | run        | epoch |    size | sha256 (first 16)  |
|----------------------------------------------------------------------------------------------------------------------------------------------|------------|------:|--------:|--------------------|
| [`sphere_K16_9ocz3qmi_epoch254.ckpt`](https://github.com/tobias-nolz/vnegnn/releases/download/v1.0-thesis/sphere_K16_9ocz3qmi_epoch254.ckpt) | `9ocz3qmi` |   254 | 14.9 MB | `4b0b0083da11a4a4` |
| [`k32_44jnpld6_epoch224.ckpt`](https://github.com/tobias-nolz/vnegnn/releases/download/v1.0-thesis/k32_44jnpld6_epoch224.ckpt)               | `44jnpld6` |   224 | 14.9 MB | `93a2840d711bfc0f` |
| [`best_zero_shot.ckpt`](https://github.com/tobias-nolz/vnegnn/releases/download/v1.0-thesis/best_zero_shot.ckpt)                             | `a5om5ixz` |  1299 | 14.8 MB | `84dbdc88fbaa53b5` |

`best_zero_shot.ckpt` is the **zero-shot baseline**: the original VN-EGNN pipeline, trained
on sc-PDB only, with no site-type head and no ASD data. It is the checkpoint behind both
zero-shot rows in [`results/asd_results.md`](results/asd_results.md).

Download into `checkpoints/` and verify against the tracked
[`../checkpoints/SHA256SUMS`](../checkpoints/SHA256SUMS):

```bash
mkdir -p checkpoints && cd checkpoints
BASE=https://github.com/tobias-nolz/vnegnn/releases/download/v1.0-thesis
curl -LO $BASE/sphere_K16_9ocz3qmi_epoch254.ckpt
curl -LO $BASE/k32_44jnpld6_epoch224.ckpt
curl -LO $BASE/best_zero_shot.ckpt
sha256sum -c SHA256SUMS
```

`SHA256SUMS` lists all three by bare filename, so `-c` passes when all three sit in
`checkpoints/`.

## What each one is for

**`sphere` (K=16) — best rank-0 localisation.** The headline localisation result: rank-0 DCC
**0.170**, the best of any configuration, against 0.140 for the matched zero-shot baseline.
Its classifier is the weaker of the two (AUROC 0.698).

**`k32` (K=32) — the deployment model.** Best ceiling and best typing: rank-7 DCC **0.240**,
rank-7 DCA **0.435**, classifier AUROC **0.834**. Its confidence ranking alone gives rank-0
DCC 0.150, *below* `sphere` — the deployment result comes from re-ranking its predictions at
inference time by `z(confidence) + z(allosteric_prob)`, which lifts rank-0 DCA to **0.268**
and reaches the DCC ceiling by rank 2 instead of rank 4.

> The re-ranking is inference-only — it is not baked into the weights. Loading this
> checkpoint and reading the confidence head gives the 0.150 row, not the deployment row.
> Reproduce the deployment numbers with `rerank_probe.py` (see below).
>
> The same re-ranking applied to `sphere` *lowers* DCA_0 to 0.203. The gain is conditional
> on the stronger classifier, so do not transplant it.

## Training hyperparameters

Both were trained with the identical hyperparameters; **K is the only difference.**

| setting                               | value                      |
|---------------------------------------|----------------------------|
| `number_of_global_nodes`              | 16 (`sphere`) / 32 (`k32`) |
| `split_suffix`                        | `mmseqs30`                 |
| `single_chain_allosteric`             | `true`                     |
| `sampling_strategy` / `sample_radius` | `fibonacci` / `true`       |
| `classification_loss_weight` (λ_cls)  | 0.5                        |
| `pos_weight` (w_p)                    | 3.2                        |
| `classification_foreground_dist`      | 8 Å                        |
| `max_center_dist`                     | 8 Å                        |
| `balance_classes` / `sampling`        | `true` / `cluster`         |
| `num_layers`, `lr`, `max_epochs`      | 5, 1e-4, 400               |

Trained on the joint sc-PDB + ASD mixture; the ASD side uses the published split.

## Using them

```bash
# sphere
python src/eval.py +data=joint data.single_chain_allosteric=true \
    ckpt_path=checkpoints/sphere_K16_9ocz3qmi_epoch254.ckpt

# k32 -- K is part of the processed-cache key, so this rebuilds the graph caches
python src/eval.py +data=joint data.single_chain_allosteric=true \
    data.graph_info.number_of_global_nodes=32 \
    ckpt_path=checkpoints/k32_44jnpld6_epoch224.ckpt

# deployment ranking, from the frozen k32 prediction file
python scripts/allosteric-sites/rerank_probe.py \
    logs/eval/runs/<dir>/predictions_allosteric.csv --strategy conf_z_plus_allo_z
```

`ckpt_path` bypasses the `wandb_run_id` lookup, so these run without W&B access. See
[`commands.md`](commands.md) for the full pipeline.



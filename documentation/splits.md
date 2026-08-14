# ASD train/valid/test split

The sequence-identity-based split of the Allosteric Database (ASD) used for **every**
training and evaluation run reported in `results/asd_results.md` and in the thesis.

The ID lists live next to this file in [`splits/`](splits/). Only PDB IDs are published here; the structures themselves come from the
[RCSB PDB](https://www.rcsb.org/) and the annotations from the
[ASD](https://mdl.shsmu.edu.cn/ASD/).

| file                                                                                             |   ids | what it is                                                                         |
|--------------------------------------------------------------------------------------------------|------:|------------------------------------------------------------------------------------|
| [`splits/train_ids_allosteric_mmseqs30`](splits/train_ids_allosteric_mmseqs30)                   | 2,241 | ASD training proteins                                                              |
| [`splits/valid_ids_allosteric_mmseqs30`](splits/valid_ids_allosteric_mmseqs30)                   |   175 | ASD validation proteins                                                            |
| [`splits/test_ids_allosteric_mmseqs30`](splits/test_ids_allosteric_mmseqs30)                     |   181 | ASD test proteins (the split as generated)                                         |
| [`splits/test_ids_allosteric_mmseqs30_evaluated`](splits/test_ids_allosteric_mmseqs30_evaluated) |   168 | the subset that actually reaches the benchmark — see [Denominators](#denominators) |

Splits are disjoint: no PDB ID appears in more than one fold.

## How it was built

`scripts/allosteric-sites/make_splits.py` pools the ASD sequences with the sc-PDB
train/valid sequences into one FASTA, tagging each record `ASD:` or `SCPDB:`, and clusters
them with MMseqs2:

```
mmseqs easy-cluster combined.fasta cluster tmp --min-seq-id 0.3 -c 0.8 --cov-mode 0
```

Whole clusters — never individual proteins — are then assigned to a fold, which gives the
two guarantees the thesis relies on:

1. No test or validation protein is homologous (≥30 % identity over ≥80 % coverage) to any
   ASD training protein.
2. Any cluster containing an sc-PDB member is forced into train. Since the joint model
   also trains on sc-PDB, an ASD protein homologous to an sc-PDB training protein would
   otherwise leak orthosteric training data into the allosteric test set. Routing it to
   train instead is harmless.

Rule 2 is why train holds 86 % of the proteins rather than the ratio you would guess from
the cluster ratios: of the 402 ASD-bearing clusters, 191 are force-routed to train, and
only the remaining free clusters are divided by ratio.

## Denominators

The test list holds 181 IDs, but the reported metrics are computed over **168 proteins**.
Thirteen fail graph construction and are written to
`data/allosteric-sites/allosteric/not_parsable_test_allo.txt` by
`src/datasets/binding_dataset.py:398`:

| cause                                                                             | ids                                                            |
|-----------------------------------------------------------------------------------|----------------------------------------------------------------|
| no `binding.npz` produced by `setup_data.py` (4)                                  | `3IJJ` `4GS9` `4TME` `5NPK`                                    |
| graph build raises under single-chain selection / the 8 Å reachability filter (9) | `3IRW` `3IWN` `3MUM` `3MUR` `3MUT` `3MUV` `3MXH` `5F1U` `6S8F` |

The drop is deterministic — the same 13 at K=8, 16 and 32 — and the surviving 168 match
the distinct `protein_name` values in `predictions_allosteric.csv` exactly, for every eval
run. Train and valid lose 34 and 3 proteins the same way, leaving 2,207 and 172.

# Commands used for Experiments

## VN-EGNN Training

Run training for VN-EGNN:

```bash
python src/train.py experiment=vnegnn
```

## Allosteric Site Prediction

### Data Preparation

PDB download and ligand extraction:

```bash
python scripts/allosteric-sites/setup_data.py \
  --asd-file data/ASD_dataset/ASD_Release_202309_AS.txt \
  --exclude-ids data/equipocket/sc-pdb/splits/train_ids_scpdb \
  --exclude-ids data/equipocket/sc-pdb/splits/valid_ids_scpdb \
  --max-diff 0 \
  --jobs 32 \
  --only-lig
````

Feature extraction:

```bash
python scripts/process_data.py \
  --jobs 32 \
  --device cuda \
  --skip-depth \
  --batch 128
```

### Evaluation

Run evaluation on a specific wandb run:

```bash
python src/eval.py \
  wandb_run_id=[wandb_run_id_from_train_run] \
  +data=allosteric
```
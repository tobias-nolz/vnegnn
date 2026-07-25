"""Pre-build the joint datamodule's InMemoryDataset caches one at a time.

This script builds each still-missing cache in isolation (the heavy sc-PDB train cache is
already on disk, so it is never loaded here), freeing memory between datasets. It reuses
the datamodule's own id-loading and dataset-construction helpers, so the resulting cache
file hashes are identical to what ``train.py`` expects. Once every ``processed/*.pt``
exists, the real run only ``torch.load``s them -- no build, no spike.

Run:  python scripts/allosteric-sites/prebuild_caches.py
"""

import gc

import hydra
from hydra import compose, initialize_config_dir

from src.datasets.joint_dataset import ALLOSTERIC, ORTHOSTERIC

CONFIG_DIR = "/home/user/vnegnn/configs"


def build(dm, ids_tuple, label, site_type, is_train):
    dataset_path, ids = ids_tuple
    print(f"\n=== building cache: {label} ({len(ids)} proteins) ===", flush=True)
    ds = dm._build_dataset(dataset_path, label, ids, site_type, is_train)
    print(f"    -> {len(ds)} graphs cached under {dataset_path}/processed", flush=True)
    del ds
    gc.collect()


def main():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=vnegnn_joint"])
    dm = hydra.utils.instantiate(cfg.data)

    # sc-PDB valid (orthosteric). sc-PDB train is intentionally skipped: its 9.6 GB cache
    # already exists and loading it here would recreate the very spike we are avoiding.
    build(dm, dm._scpdb_ids("valid"), "valid_ortho", ORTHOSTERIC, is_train=False)
    # ASD train + valid (allosteric).
    build(dm, dm._allosteric_ids("train"), "train_allo", ALLOSTERIC, is_train=True)
    build(dm, dm._allosteric_ids("valid"), "valid_allo", ALLOSTERIC, is_train=False)

    print("\nAll joint caches present. train.py will now only load them.", flush=True)


if __name__ == "__main__":
    main()

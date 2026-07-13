import logging
from pathlib import Path
from typing import Literal

from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from .binding_dataset import BindingDataModule, BindingDataset
from .utils import load_if_exists

log = logging.getLogger(__name__)

# Site-type labels used to supervise the allosteric-classification head.
ORTHOSTERIC = 0
ALLOSTERIC = 1


class JointBindingDataModule(BindingDataModule):
    """Joint orthosteric + allosteric datamodule.

    Trains on a mix of sc-PDB (orthosteric, ``site_type=0``) and the ASD training
    split (allosteric, ``site_type=1``). Both datasets are concatenated into a single
    dataloader so each mini-batch contains both classes, which is what the
    classification head needs to learn a decision boundary.

    Testing reuses the established VN-EGNN orthosteric benchmarks
    (COACH420 / HOLO4K / PDBbind2020) plus the held-out ASD allosteric test split.

    The sc-PDB / benchmark datasets live under ``root`` (as in ``BindingDataModule``),
    while the ASD dataset lives under ``allosteric_root`` -- these are separate trees,
    hence the two roots.
    """

    def __init__(
        self,
        *args,
        allosteric_root: str,
        split_suffix: str = "mmseqs30",
        allosteric_dataset_name: str = "allosteric",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.allosteric_root = Path(allosteric_root)
        self.split_suffix = split_suffix
        self.allosteric_dataset_name = allosteric_dataset_name

    # ------------------------------------------------------------------ helpers

    def _scpdb_ids(self, mode: Literal["train", "valid"]) -> tuple[Path, list[str]]:
        """Load sc-PDB ids for train/valid, applying blacklist + leakage filtering."""
        dataset_path = self.root / "sc-pdb"
        complex_names_path = (
            dataset_path / "splits" / f"{mode}_ids_{self.train_valid_split}"
        )
        with open(complex_names_path, "r") as f:
            complex_names = f.read().splitlines()

        blacklist = load_if_exists(dataset_path / "splits" / "scPDB_blacklist.txt")
        leakage = load_if_exists(dataset_path / "splits" / "scPDB_leakage.txt")
        complex_names = [
            c for c in complex_names if c not in blacklist and c not in leakage
        ]
        return dataset_path, complex_names

    def _allosteric_ids(
        self, mode: Literal["train", "valid", "test"]
    ) -> tuple[Path, list[str]]:
        """Load ASD ids for the mmseqs-based split identified by ``split_suffix``."""
        dataset_path = self.allosteric_root / self.allosteric_dataset_name
        complex_names_path = (
            dataset_path / "splits" / f"{mode}_ids_allosteric_{self.split_suffix}"
        )
        with open(complex_names_path, "r") as f:
            complex_names = f.read().splitlines()
        return dataset_path, complex_names

    def _build_dataset(
        self,
        dataset_path: Path,
        label: str,
        complex_names: list[str],
        site_type: int,
        is_train: bool,
    ) -> BindingDataset:
        return BindingDataset(
            root=dataset_path,
            label=label,
            protein_names=complex_names,
            graph_info=self.graph_info,
            global_node_subsample_size=(
                self.global_node_subsample_size if is_train else 1.0
            ),
            random_rotations=self.random_rotations,
            sampling_strategy=self.sampling_strategy,
            sample_radius=self.sample_radius if is_train else False,
            n_jobs=self.n_jobs,
            backend=self.backend,
            force_reload=self.force_reload,
            site_type=site_type,
        )

    def _wrap_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            follow_batch=self.follow_batch,
        )

    def _joint_loader(self, mode: Literal["train", "valid"]) -> DataLoader:
        is_train = mode == "train"

        scpdb_path, scpdb_ids = self._scpdb_ids(mode)
        allo_path, allo_ids = self._allosteric_ids(mode)

        scpdb_ds = self._build_dataset(
            scpdb_path, f"{mode}_ortho", scpdb_ids, ORTHOSTERIC, is_train
        )
        allo_ds = self._build_dataset(
            allo_path, f"{mode}_allo", allo_ids, ALLOSTERIC, is_train
        )
        log.info(
            "Joint %s set: %d orthosteric (sc-PDB) + %d allosteric (ASD) proteins",
            mode,
            len(scpdb_ds),
            len(allo_ds),
        )
        return self._wrap_loader(
            ConcatDataset([scpdb_ds, allo_ds]), shuffle=is_train and self.shuffle
        )

    # ------------------------------------------------------------ dataloaders

    def train_dataloader(self) -> DataLoader:
        return self._joint_loader("train")

    def val_dataloader(self) -> DataLoader:
        return self._joint_loader("valid")

    @property
    def test_dataloader_indices(self) -> dict[str, int]:
        return {
            "coach420": 0,
            "holo4k": 1,
            "pdbbind2020": 2,
            "allosteric": 3,
        }

    def test_dataloader(self) -> list[DataLoader]:
        allo_path, allo_ids = self._allosteric_ids("test")
        allo_ds = self._build_dataset(
            allo_path, "test_allo", allo_ids, ALLOSTERIC, is_train=False
        )
        return [
            self._create_dataloader("coach420", site_type=ORTHOSTERIC),
            self._create_dataloader("holo4k", site_type=ORTHOSTERIC),
            self._create_dataloader("pdbbind2020", site_type=ORTHOSTERIC),
            self._wrap_loader(allo_ds, shuffle=False),
        ]

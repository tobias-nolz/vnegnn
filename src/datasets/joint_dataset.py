import logging
from collections import Counter
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torch_geometric.loader import DataLoader

from .binding_dataset import BindingDataModule, BindingDataset
from .utils import load_if_exists

log = logging.getLogger(__name__)

# Site-type labels used to supervise the allosteric-classification head.
ORTHOSTERIC = 0
ALLOSTERIC = 1

ASD_TAG = "ASD:"
SCPDB_TAG = "SCPDB:"


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
        balance_classes: bool = True,
        sampling: Literal["dataset", "cluster"] = "cluster",
        cluster_tsv: str | None = None,
        single_chain_allosteric: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.allosteric_root = Path(allosteric_root)
        self.split_suffix = split_suffix
        self.allosteric_dataset_name = allosteric_dataset_name
        self.single_chain_allosteric = single_chain_allosteric
        self.sampling = sampling
        self.cluster_tsv = (
            Path(cluster_tsv)
            if cluster_tsv is not None
            else self.allosteric_root
            / self.allosteric_dataset_name
            / "splits"
            / "_mmseqs"
            / "cluster_cluster.tsv"
        )
        self._cluster_map_cache: dict[str, str] | None = None
        self.balance_classes = balance_classes

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
        single_chain: bool = False,
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
            max_center_dist=self.max_center_dist,
            single_chain=single_chain,
        )

    def _wrap_loader(self, dataset, shuffle: bool, sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            # A sampler and shuffle are mutually exclusive in torch's DataLoader.
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            follow_batch=self.follow_batch,
            # Without this the workers are torn down and re-forked from a ~23 GB parent
            # every epoch. torch rejects persistent workers when there are none.
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def _load_cluster_map(self) -> dict[str, str]:
        """member -> representative, from the mmseqs `*_cluster.tsv`."""
        if self._cluster_map_cache is not None:
            return self._cluster_map_cache
        mapping: dict[str, str] = {}
        if self.cluster_tsv.exists():
            with open(self.cluster_tsv) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    rep, member = line.split("\t")
                    mapping[member] = rep
        self._cluster_map_cache = mapping
        return mapping

    @staticmethod
    def _protein_names_in_order(dataset: BindingDataset) -> list[str]:
        """Protein ids in *dataset index* order.

        `BindingDataset.protein_names` is the *requested* id list; the stored graphs
        follow `raw_file_names` (an os.listdir) minus whatever failed to parse, so the
        two orders do not match. The collated store is the only faithful source.
        """
        collated = getattr(dataset, "_data", None)
        names = getattr(collated, "protein_name", None)
        if names is None:
            names = [dataset[i].protein_name for i in range(len(dataset))]
        return list(names)

    def _cluster_weights(self, dataset: BindingDataset, tag: str) -> torch.Tensor:
        """Per-sample weights that make every sequence cluster equally likely.

        Both source datasets are redundant, but not equally so: at 30% identity the ASD
        training split collapses from 2220 proteins into 294 clusters (7.6 members each,
        largest 129), while sc-PDB's 4499 give 2534 clusters (1.8 each). Weighting by
        protein therefore hands the few huge ASD families most of the allosteric
        probability mass -- the top 10 families alone drew 36% of it -- and the model
        memorises them instead of learning allosteric geometry: an 842-epoch run reached
        train/dist 1.18 A against val/dist 6.99 A, and its held-out allosteric rank-7 DCC
        was *worse* than the same setup stopped at 50 epochs (0.118 vs 0.154).

        A cluster of size `s` out of `C` clusters gets total mass `1/C`, split evenly
        across its members (`1/(C*s)` each), so the weights sum to 1 for this dataset and
        two such vectors concatenate into an even split between the two halves.

        Proteins absent from the cluster table are treated as their own singleton
        cluster, which is the conservative reading (never merge unknowns).
        """
        names = self._protein_names_in_order(dataset)
        mapping = self._load_cluster_map()

        reps: list[str] = []
        unmapped = 0
        for name in names:
            rep = mapping.get(f"{tag}{name}") or mapping.get(f"{tag}{name.upper()}")
            if rep is None:
                unmapped += 1
                rep = f"__singleton__{name}"
            reps.append(rep)

        sizes = Counter(reps)
        n_clusters = len(sizes)
        weights = torch.tensor(
            [1.0 / (n_clusters * sizes[rep]) for rep in reps], dtype=torch.double
        )
        # Kish effective sample size: how many i.i.d. draws this weighting is worth.
        # Useful as a one-glance check that the reweighting did something.
        ess = (weights.sum() ** 2) / (weights**2).sum()
        log.info(
            "%s: %d proteins -> %d clusters (largest %d, unmapped %d), "
            "effective sample size %.0f",
            tag.rstrip(":"),
            len(names),
            n_clusters,
            max(sizes.values()) if sizes else 0,
            unmapped,
            ess.item(),
        )
        return weights

    def _balanced_sampler(
        self, scpdb_ds: BindingDataset, allo_ds: BindingDataset
    ) -> WeightedRandomSampler:
        """Inverse-frequency sampler over ``ConcatDataset([scpdb, allo])``.

        With ``sampling="dataset"`` each sc-PDB sample is weighted ``1/n_ortho`` and each
        ASD sample ``1/n_allo``, so in expectation half of each batch comes from each
        *dataset*. With ``sampling="cluster"`` (the default) the mass inside each half is
        additionally spread evenly over MMseqs sequence clusters -- see
        ``_cluster_weights``.

        ``num_samples`` keeps the epoch length equal to the concatenated dataset size;
        ``replacement=True`` lets the small ASD split be revisited within an epoch.

        Either way this balances dataset membership, not the classifier's target -- ASD
        proteins carry orthosteric centers as well, so the per-virtual-node class balance
        lands near 76/24 ortho/allo rather than 50/50. See ``balance_classes``.
        """
        n_ortho, n_allo = len(scpdb_ds), len(allo_ds)

        if self.sampling == "cluster" and self.cluster_tsv.exists():
            weights = torch.cat(
                [
                    self._cluster_weights(scpdb_ds, SCPDB_TAG),
                    self._cluster_weights(allo_ds, ASD_TAG),
                ]
            )
        else:
            if self.sampling == "cluster":
                log.warning(
                    "sampling='cluster' but %s is missing; falling back to per-protein "
                    "dataset balancing. Run scripts/allosteric-sites/make_splits.py to "
                    "generate the cluster table.",
                    self.cluster_tsv,
                )
            weights = torch.cat(
                [
                    torch.full((n_ortho,), 1.0 / max(n_ortho, 1), dtype=torch.double),
                    torch.full((n_allo,), 1.0 / max(n_allo, 1), dtype=torch.double),
                ]
            )

        return WeightedRandomSampler(
            weights, num_samples=n_ortho + n_allo, replacement=True
        )

    def _joint_loader(self, mode: Literal["train", "valid"]) -> DataLoader:
        is_train = mode == "train"

        scpdb_path, scpdb_ids = self._scpdb_ids(mode)
        allo_path, allo_ids = self._allosteric_ids(mode)

        scpdb_ds = self._build_dataset(
            scpdb_path, f"{mode}_ortho", scpdb_ids, ORTHOSTERIC, is_train
        )
        allo_ds = self._build_dataset(
            allo_path,
            f"{mode}_allo",
            allo_ids,
            ALLOSTERIC,
            is_train,
            single_chain=self.single_chain_allosteric,
        )
        log.info(
            "Joint %s set: %d orthosteric (sc-PDB) + %d allosteric (ASD) proteins",
            mode,
            len(scpdb_ds),
            len(allo_ds),
        )
        concat = ConcatDataset([scpdb_ds, allo_ds])

        if is_train and self.balance_classes:
            sampler = self._balanced_sampler(scpdb_ds, allo_ds)
            log.info(
                "Training loader uses balanced sampling (~50/50 ortho/allo, "
                "sampling=%s).",
                self.sampling,
            )
            return self._wrap_loader(concat, shuffle=False, sampler=sampler)

        return self._wrap_loader(concat, shuffle=is_train and self.shuffle)

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
            allo_path,
            "test_allo",
            allo_ids,
            ALLOSTERIC,
            is_train=False,
            single_chain=self.single_chain_allosteric,
        )
        return [
            self._create_dataloader("coach420", site_type=ORTHOSTERIC),
            self._create_dataloader("holo4k", site_type=ORTHOSTERIC),
            self._create_dataloader("pdbbind2020", site_type=ORTHOSTERIC),
            self._wrap_loader(allo_ds, shuffle=False),
        ]

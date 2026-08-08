import json
import logging
import os
from hashlib import sha256
from pathlib import Path
from typing import Literal

import lightning as pl
import numpy as np
import torch
from einops import repeat
from joblib import Parallel, cpu_count, delayed
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from src.utils.graph import sample_fibonacci_grid, sample_uniform_in_sphere

from .utils import GraphInfo, create_hetero_graph, load_if_exists, to_serializable

log = logging.getLogger(__name__)

TEST_DATALOADER_INDICES = {
    "coach420": 0,
    "holo4k": 1,
    "pdbbind2020": 2,
    "sc-pdb": 3,
}

DEFAULT_MAX_CENTER_DIST = 8.0
DEFAULT_HOST_CONTACT_DIST = 10.0


def _apply_site_keep_mask(
    keep: np.ndarray,
    binding_sites: np.ndarray,
    ligand_coords: np.ndarray,
    ligand_ids: np.ndarray,
    site_types: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Keep the sites flagged by boolean ``keep`` (one entry per site), renumbering
    ``ligand_ids`` to stay contiguous 0..k-1 indices into the survivors and dropping the
    ligand atoms of removed sites. Shared by ``drop_unreachable_sites`` and
    ``select_single_chain`` so the renumbering logic lives in one place.
    """
    # Old site index -> new index; -1 marks a dropped site.
    remap = np.full(len(keep), -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))

    atom_keep = keep[ligand_ids]
    ligand_coords = ligand_coords[atom_keep]
    ligand_ids = remap[ligand_ids[atom_keep]]
    binding_sites = binding_sites[keep]
    if site_types is not None:
        site_types = np.asarray(site_types)[keep]
    return binding_sites, ligand_coords, ligand_ids, site_types


def drop_unreachable_sites(
    protein_name: str,
    coords: np.ndarray,
    binding_sites: np.ndarray,
    ligand_coords: np.ndarray,
    ligand_ids: np.ndarray,
    site_types: np.ndarray | None,
    max_center_dist: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Remove binding sites whose center lies further than `max_center_dist` from the
    protein, renumbering `ligand_ids` so they stay contiguous indices into the sites.

    NaN centers (a ligand with no protein atom inside the binding threshold, which
    extract_binding_info turns into a mean over an empty slice) fail the comparison and
    are dropped here too. That is a deliberate upgrade: `create_hetero_graph` rejects the
    *whole protein* if any center is NaN, which cost 44 ASD training, 15 validation and 5
    test proteins over a handful of bad sites each -- none of them all-NaN.

    Returns the filtered ``(binding_sites, ligand_coords, ligand_ids, site_types)``.
    Raises if nothing survives, which `process()` turns into a dropped protein.
    """
    if binding_sites is None or len(binding_sites) == 0:
        return binding_sites, ligand_coords, ligand_ids, site_types

    dists = np.linalg.norm(
        binding_sites[:, None, :] - coords[None, :, :], axis=-1
    ).min(axis=1)
    # `nan <= x` is False, so NaN centers land in the dropped set.
    keep = dists <= max_center_dist
    if keep.all():
        return binding_sites, ligand_coords, ligand_ids, site_types

    n_nan = int(np.isnan(dists).sum())
    if not keep.any():
        raise ValueError(
            f"All {len(binding_sites)} binding-site centers of {protein_name} are "
            f"unusable ({n_nan} NaN, rest >{max_center_dist} A from the parsed "
            f"structure, min {np.nanmin(dists):.1f} A) -- the annotation does not "
            f"match res_coords."
        )

    log.warning(
        "%s: dropping %d/%d binding site(s) -- %d NaN, %d further than %.1f A from any "
        "residue (furthest %.1f A)",
        protein_name,
        int((~keep).sum()),
        len(keep),
        n_nan,
        int((~keep).sum()) - n_nan,
        max_center_dist,
        float(np.nanmax(dists)),
    )

    return _apply_site_keep_mask(
        keep, binding_sites, ligand_coords, ligand_ids, site_types
    )


def _site_host_chains(
    coords: np.ndarray,
    chains: np.ndarray,
    ligand_coords: np.ndarray,
    ligand_ids: np.ndarray,
    n_sites: int,
    host_contact_dist: float,
) -> tuple[list[str], np.ndarray]:
    """For each of the ``n_sites`` sites, the chain that contributes the most CA residues
    within ``host_contact_dist`` of that site's ligand atoms (its 'host'), plus that contact
    count. Ties resolve to the lexicographically smallest chain id (``np.unique`` sorts),
    so the choice is deterministic. Sites whose ligand has no CA within reach fall back to
    the chain of the single nearest CA; a site with no ligand atoms at all gets host "" (an
    id no real chain carries), which makes it droppable. Indexing by site id (not by
    ``unique(ligand_ids)``) keeps ``hosts``/``counts`` aligned with ``binding_sites``.
    """
    hosts: list[str] = []
    counts = np.zeros(n_sites, dtype=np.int64)
    for i in range(n_sites):
        latoms = ligand_coords[ligand_ids == i]
        if latoms.size == 0:
            hosts.append("")  # empty id matches no real chain -> site is droppable
            continue
        nearest = np.linalg.norm(
            coords[:, None, :] - latoms[None, :, :], axis=-1
        ).min(axis=1)
        contact = nearest <= host_contact_dist
        if contact.any():
            uniq, cnt = np.unique(chains[contact], return_counts=True)
            hosts.append(str(uniq[int(np.argmax(cnt))]))
            counts[i] = int(cnt.max())
        else:
            hosts.append(str(chains[int(np.argmin(nearest))]))
            counts[i] = 1
    return hosts, counts


def select_single_chain(
    protein_name: str,
    coords: np.ndarray,
    chains: np.ndarray,
    binding_sites: np.ndarray,
    ligand_coords: np.ndarray,
    ligand_ids: np.ndarray,
    site_types: np.ndarray | None,
    host_contact_dist: float = DEFAULT_HOST_CONTACT_DIST,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict a (possibly multimeric) ASD protein to the single chain that hosts its
    allosteric pocket.

    The deposited ASD structure is the full assembly (median 723 residues, 76%
    multi-chain), which spreads the fixed K=8 virtual nodes over a huge Fibonacci sphere
    and, for homo-multimers, replicates the same pocket on every protomer. This picks the
    chain hosting the best-resolved *allosteric* site -- ground truth, derived from the
    annotated modulator ligand's residue contacts, never from model output -- keeps only
    that chain, and keeps only the sites hosted by it. Symmetry-replicated copies on other
    chains fall away, restoring sc-PDB-like single-chain / single-pocket-region geometry.

    Returns ``(residue_keep, site_keep)`` boolean masks over ``coords`` and the sites. The
    caller subsets the per-residue arrays with ``residue_keep`` and hands ``site_keep`` to
    ``_apply_site_keep_mask``. The chosen anchor site is always kept, so neither mask is
    ever all-False.
    """
    n_res = len(coords)
    n_sites = 0 if binding_sites is None else len(binding_sites)
    if n_res == 0 or n_sites == 0:
        return np.ones(n_res, dtype=bool), np.ones(n_sites, dtype=bool)

    hosts, counts = _site_host_chains(
        coords, chains, ligand_coords, ligand_ids, n_sites, host_contact_dist
    )

    # Anchor on an allosteric site when the per-site labels are present; ASD-without-ortho
    # proteins omit site_types (every site is allosteric), so fall back to all sites.
    if site_types is not None and (np.asarray(site_types) == 1).any():
        candidates = np.flatnonzero(np.asarray(site_types) == 1)
    else:
        candidates = np.arange(n_sites)

    primary = int(candidates[int(np.argmax(counts[candidates]))])
    anchor = hosts[primary]

    residue_keep = chains == anchor
    site_keep = np.array([h == anchor for h in hosts], dtype=bool)
    return residue_keep, site_keep


class BindingDataset(InMemoryDataset):
    def __init__(
        self,
        root: Path,
        protein_names: list[str],
        graph_info: GraphInfo,
        label="train",
        n_jobs: int = cpu_count() - 1,
        backend: str = "loky",
        random_rotations: bool = False,
        global_node_subsample_size: float = 1.0,
        sampling_strategy: Literal["fibonacci", "uniform"] = "fibonacci",
        sample_radius: bool = False,
        force_reload: bool = False,
        site_type: int = 0,
        max_center_dist: float | None = DEFAULT_MAX_CENTER_DIST,
        single_chain: bool = False,
        per_site_labels: bool = True,
        drop_ortho_augmentation: bool = False,
    ):
        self.protein_names = protein_names
        self.graph_info = graph_info
        self.label = label
        self.n_jobs = n_jobs
        self.backend = backend

        self.random_rotations = random_rotations
        self.global_node_subsample_size = global_node_subsample_size
        self.sampling_strategy = sampling_strategy
        self.sample_radius = sample_radius
        self.site_type = site_type
        self.max_center_dist = max_center_dist
        self.single_chain = single_chain
        # Label-scheme ablations (Section 5.5, tab:classification-results). Both default to
        # the deployed behaviour, so the cache key and graphs are unchanged when unset.
        #   per_site_labels=False       -> per-protein labels: every center inherits the
        #                                  dataset-level `site_type` instead of its own class.
        #   drop_ortho_augmentation=True -> no --extract-orthosteric augmentation: drop the
        #                                  class-0 centers harvested from ASD proteins.
        self.per_site_labels = per_site_labels
        self.drop_ortho_augmentation = drop_ortho_augmentation

        super().__init__(root, force_reload=force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        raw_file_names = os.listdir(self.raw_dir)
        if len(self.protein_names[0]) != len(raw_file_names[0]):
            return [f for f in raw_file_names if f[:4] in self.protein_names]
        return [f for f in raw_file_names if f in self.protein_names]

    @property
    def processed_file_names(self):
        graph_info_ha = json.dumps(to_serializable(self.graph_info), sort_keys=True)
        protein_ha = json.dumps(to_serializable(self.protein_names), sort_keys=True)

        graph_info_ha = sha256(graph_info_ha.encode()).hexdigest()
        protein_ha = sha256(protein_ha.encode()).hexdigest()
        site_type_ha = sha256(str(self.site_type).encode()).hexdigest()
        center_dist_ha = sha256(str(self.max_center_dist).encode()).hexdigest()
        hash_parts = graph_info_ha + protein_ha + site_type_ha + center_dist_ha
        # Only perturb the hash when single-chain is on, so existing full-assembly caches
        # stay valid: single_chain=False reproduces the pre-Route-A cache key exactly.
        if self.single_chain:
            hash_parts += sha256(b"single_chain").hexdigest()
        # The label-scheme ablations change the cached `bindingsite_site_type` (and, for the
        # augmentation drop, the center set), so they must fork the cache. Defaults leave the
        # key untouched.
        if not self.per_site_labels:
            hash_parts += sha256(b"per_protein_labels").hexdigest()
        if self.drop_ortho_augmentation:
            hash_parts += sha256(b"drop_ortho_aug").hexdigest()
        full_hash = sha256(hash_parts.encode()).hexdigest()[:8]

        name = f"{full_hash}_{self.label}.pt"
        return [name]

    def process(self):
        def process_protein(path: Path):
            try:
                binding_info = np.load(path / "binding.npz")
                coords = binding_info["res_coords"]
                ligand_coords = binding_info["ligand_coords"]
                ligand_ids = binding_info["ligand_ids"]
                res_names = binding_info["res_names"]
                binding_residues = binding_info["binding_residues"]
                binding_sites = binding_info["binding_site_centers"]
                esm_features = np.load(path / "embeddings.npz")["residue_embeddings"]
                res_depths = binding_info["res_depths"]
                # Per-center labels are only present for mixed (ortho+allo) proteins;
                # otherwise fall back to the dataset-level `site_type`.
                site_types = (
                    binding_info["site_types"]
                    if "site_types" in binding_info.files
                    else None
                )

                if (
                    self.drop_ortho_augmentation
                    and self.site_type == 1  # 1 == allosteric (see joint_dataset.ALLOSTERIC)
                    and site_types is not None
                ):
                    keep_allo = np.asarray(site_types) != 0  # 0 == orthosteric
                    if keep_allo.any() and not keep_allo.all():
                        (
                            binding_sites,
                            ligand_coords,
                            ligand_ids,
                            site_types,
                        ) = _apply_site_keep_mask(
                            keep_allo,
                            binding_sites,
                            ligand_coords,
                            ligand_ids,
                            site_types,
                        )

                if self.single_chain:
                    residue_keep, site_keep = select_single_chain(
                        protein_name=path.stem,
                        coords=coords,
                        chains=binding_info["chains"],
                        binding_sites=binding_sites,
                        ligand_coords=ligand_coords,
                        ligand_ids=ligand_ids,
                        site_types=site_types,
                    )
                    coords = coords[residue_keep]
                    res_names = res_names[residue_keep]
                    binding_residues = binding_residues[residue_keep]
                    esm_features = esm_features[residue_keep]
                    res_depths = res_depths[residue_keep]
                    (
                        binding_sites,
                        ligand_coords,
                        ligand_ids,
                        site_types,
                    ) = _apply_site_keep_mask(
                        site_keep, binding_sites, ligand_coords, ligand_ids, site_types
                    )

                if self.max_center_dist is not None:
                    (
                        binding_sites,
                        ligand_coords,
                        ligand_ids,
                        site_types,
                    ) = drop_unreachable_sites(
                        protein_name=path.stem,
                        coords=coords,
                        binding_sites=binding_sites,
                        ligand_coords=ligand_coords,
                        ligand_ids=ligand_ids,
                        site_types=site_types,
                        max_center_dist=self.max_center_dist,
                    )

                graph_site_types = site_types if self.per_site_labels else None

                return create_hetero_graph(
                    protein_name=path.stem,
                    coords=coords,
                    ligand_coords=ligand_coords,
                    ligand_ids=ligand_ids,
                    res_names=res_names,
                    res_depths=res_depths,
                    binding_sites=binding_sites,
                    binding_residues=binding_residues,
                    esm_features=esm_features,
                    graph_info=self.graph_info,
                    site_type=self.site_type,
                    site_types=graph_site_types,
                )
            except Exception as e:
                log.warning(f"Error in {path}: {e}")
                return path.stem

        log.info(
            "Starting parallel protein-to-graph conversion for %d proteins, using backend=%s with %d jobs",
            len(self.raw_file_names),
            self.backend,
            self.n_jobs,
        )
        results = Parallel(n_jobs=self.n_jobs, verbose=1, timeout=None, backend=self.backend)(
            delayed(process_protein)(Path(f"{self.raw_dir}/{file_name}"))
            for file_name in tqdm(
                self.raw_file_names,
                desc=f"build_graphs[{self.label}]",
            )
        )

        log.info("Finished proteins to graph.")

        not_parsable = [g for g in results if isinstance(g, str)]
        if len(not_parsable) > 0:
            log.info(
                "Write non parsable complexes to file (%d entries)",
                len(not_parsable),
            )
            with open(self.root / f"not_parsable_{self.label}.txt", "w") as f:
                f.write("\n".join(not_parsable))

        data_graphs = [g for g in results if not isinstance(g, str)]
        data, slices = self.collate(data_graphs)
        torch.save((data, slices), self.processed_paths[0])
        log.info("Finished saving data to %s", self.processed_paths[0])

    def get(self, idx: int) -> Data:
        graph = super().get(idx)
        num_points = graph["global_node"].pos.shape[0]

        if self.sampling_strategy == "fibonacci":
            radius = graph.radius
            if self.sample_radius:
                radius = torch.rand(1) * radius
            graph["global_node"].pos = sample_fibonacci_grid(
                graph.centroid,
                radius,
                num_points,
                random_rotations=self.random_rotations,
            )
        elif self.sampling_strategy == "center":
            centers = repeat(graph["atom"].pos.mean(dim=0), "d -> n d", n=num_points)
            graph["global_node"].pos = centers
            graph["global_node"].x = graph["global_node"].x + torch.randn_like(
                graph["global_node"]["x"]
            )
        else:
            graph["global_node"].pos = sample_uniform_in_sphere(
                graph.centroid, graph.radius, num_points
            )

        if self.global_node_subsample_size < 1.0:
            global_node_edge_index = graph[
                "global_node", "to", "atom"
            ].edge_index  # noqa: F821
            new_edge_index = self.random_edge_subset(
                global_node_edge_index, self.global_node_subsample_size
            )
            graph["global_node", "to", "atom"].edge_index = torch.stack(
                new_edge_index
            )  # noqa: F821
            graph["atom", "to", "global_node"].edge_index = torch.stack(
                [new_edge_index[1], new_edge_index[0]]
            )

        graph["atom"].x = graph["atom"].x.float()
        graph["atom"].y = graph["atom"].y.float()
        graph["atom"].pos = graph["atom"].pos.float()
        graph["atom"].bindingsite_center = graph["atom"].bindingsite_center.float()

        graph["global_node"].x = graph["global_node"].x.float()
        graph["global_node"].pos = graph["global_node"].pos.float()

        graph["ligand"].ligand_coords = graph["ligand"].ligand_coords.float()
        graph["ligand"].ligand_ids = graph["ligand"].ligand_ids.float()
        graph["atom"].res_depths = graph["atom"].res_depths.float()

        return graph

    @staticmethod
    def random_edge_subset(edge_index: torch.Tensor, subsample_size: float):
        """Randomly subsample edges from edge_index.

        Args:
            edge_index: Edge index tensor of shape [2, num_edges]
            subsample_size: Fraction of edges to keep (0 to 1)

        Returns:
            Tuple of (src, dst) edge lists
        """
        num_edges = edge_index.shape[1]
        num_keep = int(num_edges * subsample_size)

        # Sample random indices
        perm = torch.randperm(num_edges)[:num_keep]

        # Select edges
        new_edge_index = edge_index[:, perm]

        return new_edge_index[0], new_edge_index[1]


class BindingDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root: str,
        graph_info: GraphInfo,
        global_node_subsample_size: float = 1.0,
        random_rotations: bool = True,
        sampling_strategy: str = "fibonacci",
        sample_radius: bool = False,
        train_valid_split: float = 0,
        n_jobs: int = cpu_count() - 1,
        backend: str = "loky",
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 10,
        force_reload: bool = False,
        max_center_dist: float | None = DEFAULT_MAX_CENTER_DIST,
        follow_batch: list[str] = [
            "ligand",
            "ligand_coords",
            "bindingsite_center",
            "ligand_ids",
        ],
    ):
        super().__init__()
        self.max_center_dist = max_center_dist
        self.root = Path(root)
        self.graph_info = graph_info
        self.global_node_subsample_size = global_node_subsample_size
        self.random_rotations = random_rotations
        self.sampling_strategy = sampling_strategy
        self.sample_radius = sample_radius
        self.train_valid_split = train_valid_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.force_reload = force_reload
        self.follow_batch = follow_batch
        self.n_jobs = n_jobs
        self.backend = backend

    def _create_dataloader(
        self,
        mode: Literal["train", "valid", "coach420", "holo4k", "pdbbind2020", "allosteric"],
        site_type: int = 0,
    ) -> DataLoader:
        match mode:
            case "train" | "valid":
                dataset_path = self.root / "sc-pdb"
                complex_names_path = (
                    dataset_path / "splits" / f"{mode}_ids_{self.train_valid_split}"
                )
                with open(complex_names_path, "r") as f:
                    complex_names = f.read().splitlines()

                blacklist_path = dataset_path / "splits" / "scPDB_blacklist.txt"
                leakage_path = dataset_path / "splits" / "scPDB_leakage.txt"
                blacklist = load_if_exists(blacklist_path)
                leakage = load_if_exists(leakage_path)

                complex_names = [
                    c for c in complex_names if c not in blacklist and c not in leakage
                ]
            case "coach420" | "holo4k" | "pdbbind2020" | "allosteric":
                dataset_path = self.root / mode
                complex_names_path = dataset_path / "splits" / f"test_ids_{mode}"
                with open(complex_names_path, "r") as f:
                    complex_names = f.read().splitlines()
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        return DataLoader(
            BindingDataset(
                root=dataset_path,
                label=mode,
                protein_names=complex_names,
                graph_info=self.graph_info,
                global_node_subsample_size=(
                    self.global_node_subsample_size if mode == "train" else 1.0
                ),
                random_rotations=self.random_rotations,
                sampling_strategy=self.sampling_strategy,
                sample_radius=self.sample_radius if mode == "train" else False,
                n_jobs=self.n_jobs,
                backend=self.backend,
                force_reload=self.force_reload,
                site_type=site_type,
                max_center_dist=self.max_center_dist,
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle if mode == "train" else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            follow_batch=self.follow_batch,
        )

    def train_dataloader(self):
        return self._create_dataloader("train")

    def val_dataloader(self):
        return self._create_dataloader("valid")

    def test_dataloader(self):
        return [
            self._create_dataloader("coach420"),
            self._create_dataloader("holo4k"),
        ]


class BindingEpDataModule(BindingDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def test_dataloader_indices(self) -> dict[str, int]:
        return {
            "coach420": 0,
            "holo4k": 1,
            "pdbbind2020": 2,
        }

    def test_dataloader(self):
        return [
            self._create_dataloader("coach420"),
            self._create_dataloader("holo4k"),
            self._create_dataloader("pdbbind2020"),
        ]


class BindingPDBTrainDataModule(BindingDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_dataloader(
        self, mode: Literal["train", "valid", "coach420", "holo4k", "sc-pdb"]
    ) -> DataLoader:
        match mode:
            case "train" | "valid":
                dataset_path = self.root / "pdbbind2020"
                complex_names_path = (
                    dataset_path / "splits" / f"{mode}_ids_{self.train_valid_split}"
                )
                with open(complex_names_path, "r") as f:
                    complex_names = f.read().splitlines()

            case "coach420" | "holo4k" | "sc-pdb":
                dataset_path = self.root / mode
                complex_names_path = (
                    dataset_path / "splits" / f"test_ids_{mode}_pdbbind2020"
                )
                with open(complex_names_path, "r") as f:
                    complex_names = f.read().splitlines()
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        return DataLoader(
            BindingDataset(
                root=dataset_path,
                label=mode,
                protein_names=complex_names,
                graph_info=self.graph_info,
                global_node_subsample_size=(
                    self.global_node_subsample_size if mode == "train" else 1.0
                ),
                random_rotations=self.random_rotations,
                sampling_strategy=self.sampling_strategy,
                sample_radius=self.sample_radius if mode == "train" else False,
                n_jobs=self.n_jobs,
                force_reload=self.force_reload,
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle if mode == "train" else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            follow_batch=self.follow_batch,
        )

    @property
    def test_dataloader_indices(self) -> dict[str, int]:
        return {
            "coach420": 0,
            "holo4k": 1,
            "sc-pdb": 2,
        }

    def train_dataloader(self):
        return self._create_dataloader("train")

    def val_dataloader(self):
        return self._create_dataloader("valid")

    def test_dataloader(self):
        return [
            self._create_dataloader("coach420"),
            self._create_dataloader("holo4k"),
            self._create_dataloader("sc-pdb"),
        ]

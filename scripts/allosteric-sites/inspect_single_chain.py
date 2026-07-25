#!/usr/bin/env python
"""
Applies `select_single_chain` (the exact function the dataset uses) to every ASD raw
`binding.npz` and reports how the graph geometry shifts, WITHOUT building any processed
cache. Run this before flipping `single_chain_allosteric: true` and rebuilding, to confirm
the reduction moves the numbers the intended way:

  - node count and Fibonacci-sphere radius drop (memory peak + density-collapse fix),
  - sites/protein and %multi-chain fall (symmetry-replicated pockets collapsed).

The plan's 15.7 A (ASD) / 3.6 A (coach420) virtual-node -> center distances are a
*post-inference* metric (the model's moved global-node positions, `vn_center_dist`) and
cannot be reproduced by a static script; the eval reports those on the trained model. The
model-independent stand-in here is the sphere radius, since with K fixed at 8 the initial
coverage density goes as K / radius**2.

Usage:
  python scripts/allosteric-sites/inspect_single_chain.py \
      [--raw-dir data/allosteric-sites/allosteric/raw] [--show 15]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Repo root on sys.path so `src...` imports resolve when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.binding_dataset import (  # noqa: E402
    _apply_site_keep_mask,
    drop_unreachable_sites,
    select_single_chain,
)

MAX_CENTER_DIST = 8.0


def sphere_radius(coords: np.ndarray) -> float:
    """The virtual-node Fibonacci-sphere radius the dataset assigns this graph:
    ``max||coord - centroid||`` (create_hetero_graph). With K fixed at 8, initial coverage
    density goes as K/radius**2, so this is the model-independent density-collapse indicator."""
    if len(coords) == 0:
        return float("nan")
    return float(np.max(np.linalg.norm(coords - coords.mean(axis=0), axis=1)))


def reduce_sites(coords, binding_sites, ligand_coords, ligand_ids, site_types, name):
    """Mirror the pipeline's max_center_dist drop; return None if nothing survives."""
    try:
        return drop_unreachable_sites(
            name, coords, binding_sites, ligand_coords, ligand_ids, site_types, MAX_CENTER_DIST
        )
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/allosteric-sites/allosteric/raw")
    ap.add_argument("--show", type=int, default=15, help="list this many biggest reducers")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    dirs = sorted(p for p in raw.iterdir() if (p / "binding.npz").exists())
    print(f"Scanning {len(dirs)} ASD proteins under {raw}\n")

    rows = []  # (name, res_b, res_a, ch_b, ch_a, sites_b, sites_a, rad_b, rad_a)
    n_multichain = n_shrunk = n_dropped_before = n_dropped_after = 0

    for d in dirs:
        b = np.load(d / "binding.npz")
        coords = b["res_coords"]
        chains = b["chains"]
        centers = b["binding_site_centers"]
        lcoords = b["ligand_coords"]
        lids = b["ligand_ids"]
        stypes = b["site_types"] if "site_types" in b.files else None
        n_ch = len(np.unique(chains))
        if n_ch > 1:
            n_multichain += 1

        # BEFORE: full assembly + the existing max_center_dist drop.
        before = reduce_sites(coords, centers, lcoords, lids, stypes, d.name)
        if before is None:
            n_dropped_before += 1
            continue
        c_b, _, _, _ = before

        # AFTER: single-chain mask, then the same max_center_dist drop on the reduced graph.
        rkeep, skeep = select_single_chain(d.name, coords, chains, centers, lcoords, lids, stypes)
        coords_a = coords[rkeep]
        centers_a, lcoords_a, lids_a, stypes_a = _apply_site_keep_mask(
            skeep, centers, lcoords, lids, stypes
        )
        after = reduce_sites(coords_a, centers_a, lcoords_a, lids_a, stypes_a, d.name)
        if after is None:
            n_dropped_after += 1
            continue
        c_a, _, _, _ = after

        if len(coords_a) < len(coords):
            n_shrunk += 1

        rows.append(
            (d.name, len(coords), len(coords_a), n_ch, len(np.unique(chains[rkeep])),
             len(c_b), len(c_a), sphere_radius(coords), sphere_radius(coords_a))
        )

    if not rows:
        print("No proteins survived -- check the raw dir.")
        return

    arr = np.array([r[1:] for r in rows], dtype=float)
    res_b, res_a, _, _, sites_b, sites_a, rad_b, rad_a = arr.T

    def med(x):
        return float(np.nanmedian(x))

    print(f"proteins kept          : {len(rows)}  "
          f"(multi-chain {n_multichain}, shrunk {n_shrunk}, "
          f"dropped before/after {n_dropped_before}/{n_dropped_after})")
    print(f"residues/protein median: {med(res_b):7.0f}  ->  {med(res_a):7.0f}")
    print(f"sites/protein   median : {med(sites_b):7.1f}  ->  {med(sites_a):7.1f}")
    print(f"VN sphere radius median: {med(rad_b):7.1f}  ->  {med(rad_a):7.1f}  A"
          f"   (K=8 density ~ K/r^2 -> x{(med(rad_b)/med(rad_a))**2:.1f} denser)")
    print(f"total residues         : {int(res_b.sum()):>8}  ->  {int(res_a.sum()):>8}"
          f"   ({100*(1-res_a.sum()/res_b.sum()):.0f}% fewer)")
    print(f"largest graph (nodes)  : {int(res_b.max()):>8}  ->  {int(res_a.max()):>8}"
          f"   (sets the memory peak)")

    print("\nBiggest reducers (by residues removed):")
    print(f"  {'pdb':6} {'res':>13} {'chains':>9} {'sites':>9}  {'radius A':>13}")
    for r in sorted(rows, key=lambda r: r[2] - r[1])[: args.show]:
        name, rb, ra, cb, ca, sb, sa, radb, rada = r
        print(f"  {name:6} {rb:5d}->{ra:5d} {cb:4d}->{ca:2d} {sb:4d}->{sa:2d}"
              f"     {radb:5.1f}->{rada:5.1f}")


if __name__ == "__main__":
    main()

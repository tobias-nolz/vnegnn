#!/usr/bin/env python
"""No-training re-rank probe of the allosteric DCC/DCA ranking gap.

The DCC0-vs-DCC7 (and much larger DCA0-vs-DCA7) gap is a *ranking* failure: with
K=16..32 virtual nodes the model lands a node near the ligand often (high DCC7/DCA7
ceiling) but the confidence head surfaces the wrong node first (low DCC0/DCA0). This
script tests -- with zero model inference -- whether re-ordering the *same* predicted
clusters by an alternative scalar closes that gap. Two families are available: geometry
priors derived from the residue cloud, and classifier-driven scores that promote nodes
the model calls allosteric over the (usually higher-confidence) orthosteric pocket.

It reuses the exact eval scoring path (`evaluate_all_proteins`), only swapping the
ranking scalar via `rank_by`. The `confidence_0` row must reproduce the logged wandb
numbers -- that is the harness-correctness check before any other row is trusted.

Examples:
    # score one finished evaluation (K is read from the file)
    python scripts/allosteric-sites/rerank_probe.py logs/eval/runs/<dir>/predictions_allosteric.csv

    # several at once, with labels
    python scripts/allosteric-sites/rerank_probe.py \\
        --label sphere logs/eval/runs/<dir_a>/predictions_allosteric.csv \\
        --label K32    logs/eval/runs/<dir_b>/predictions_allosteric.csv

    # only the deployment ranking, against the baseline
    python scripts/allosteric-sites/rerank_probe.py <csv> --strategy conf_z_plus_allo_z
"""
from pathlib import Path

import click
import pandas as pd
import rootutils

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.misc import compute_metric_ratios, evaluate_all_proteins  # noqa: E402

# Anchored to the project root, so the script runs from any working directory.
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "allosteric-sites" / "allosteric" / "raw"

# `confidence_0` reproduces the logged evaluation exactly and must stay first: every
# other row is reported as a delta against it.
STRATEGIES = [
    "confidence_0",           # baseline -- must reproduce the logged numbers
    "density",                # # residues within 8A of the node (pure geometry)
    "density_close",          # # residues within 4A (immediate contacts)
    "neg_min_res_dist",       # -dist to nearest residue (penalise solvent-floaters)
    "conf_z_plus_density_z",  # confidence blended with density (z-scored)
    "conf_z_plus_negmind_z",  # confidence blended with proximity
    "conf_z_plus_densityclose_z",
    # classifier-driven: promote nodes the model calls allosteric over the (usually
    # higher-confidence) orthosteric pocket -- the natural fix on the allo-only set.
    "allo_prob",              # rank purely by allosteric probability
    "conf_z_plus_allo_z",     # confidence blended with allosteric prob (deployment)
    "conf_x_allo",            # confidence * allosteric prob
    "allo_z_plus_density_z",  # allosteric prob blended with geometry
]


def infer_num_global_nodes(df: pd.DataFrame) -> int:
    """K from the prediction file: one row per virtual node per protein.

    Every protein contributes exactly K rows, so the per-protein row count *is* K. A
    ragged file means the CSV is truncated or concatenated from runs with different K,
    which would silently mis-size the rank columns, so refuse it rather than guess.
    """
    counts = df.groupby("protein_name").size()
    if counts.nunique() != 1:
        raise click.ClickException(
            f"cannot infer K: proteins have {counts.min()}..{counts.max()} rows. "
            "Pass --num-global-nodes explicitly."
        )
    return int(counts.iloc[0])


def rank_curve(df, protein_path, k, rank_by, site_type_filter, max_center_dist):
    res = evaluate_all_proteins(
        df=df,
        protein_path=protein_path,
        num_global_nodes=k,
        site_type_filter=site_type_filter,
        max_center_dist=max_center_dist,
        rank_by=rank_by,
        show_progress=False,
    )
    df_res = pd.DataFrame(res)
    _, dca, _ = compute_metric_ratios(df_res, "dca")
    _, dcc, _ = compute_metric_ratios(df_res, "dcc")
    dca_r = [dca[f"dca_ratio_n_rank_dca_{i}"].mean() for i in range(k)]
    dcc_r = [dcc[f"dcc_ratio_n_rank_dcc_{i}"].mean() for i in range(k)]
    return len(df_res), dca_r, dcc_r


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "predictions", nargs=-1, required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--label", "labels", multiple=True,
    help="Display name per prediction file, in order. Defaults to the eval directory name.",
)
@click.option(
    "--num-global-nodes", "-k", type=int, default=None,
    help="K. Inferred from each file's rows-per-protein when omitted.",
)
@click.option(
    "--raw-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_RAW_DIR, show_default=True,
    help="ASD raw/ folder holding the per-protein binding.npz ground truth.",
)
@click.option(
    "--site-type-filter", type=int, default=1, show_default=True,
    help="Restrict ground truth to this site type (1=allosteric, 0=orthosteric, -1=all).",
)
@click.option(
    "--max-center-dist", type=float, default=8.0, show_default=True,
    help="Drop ground-truth centres further than this from any parsed residue.",
)
@click.option(
    "--strategy", "strategies", multiple=True,
    help="Ranking scalar(s) to score. Repeatable; defaults to all. "
         "`confidence_0` is always included as the reproduction check.",
)
@click.option(
    "--rank", "report_rank", type=int, default=7, show_default=True,
    help="Ceiling rank to report alongside rank-0.",
)
def main(predictions, labels, num_global_nodes, raw_dir, site_type_filter,
         max_center_dist, strategies, report_rank):
    """Re-score frozen prediction files under alternative ranking scalars.

    Coordinates, ground-truth sites and the denominator are held fixed; only the order
    of each protein's clustered predictions changes.
    """
    if labels and len(labels) != len(predictions):
        raise click.ClickException(
            f"got {len(labels)} --label values for {len(predictions)} prediction files"
        )
    chosen = list(strategies) or STRATEGIES
    unknown = [s for s in chosen if s not in STRATEGIES]
    if unknown:
        raise click.ClickException(
            f"unknown strategy {unknown}; choose from {', '.join(STRATEGIES)}"
        )
    if "confidence_0" not in chosen:
        chosen = ["confidence_0"] + chosen

    for i, csv_path in enumerate(predictions):
        df = pd.read_csv(csv_path)
        k = num_global_nodes or infer_num_global_nodes(df)
        if report_rank >= k:
            raise click.ClickException(
                f"--rank {report_rank} is out of range for K={k} (max {k - 1})"
            )
        label = labels[i] if labels else csv_path.parent.name

        print("\n" + "=" * 92)
        print(f"{label}   (K={k})   {csv_path}")
        print("=" * 92)
        header = (
            f"{'strategy':<28} "
            f"{'DCA0':>6} {f'DCA{report_rank}':>6} {'DCAgap':>7} | "
            f"{'DCC0':>6} {f'DCC{report_rank}':>6} {'DCCgap':>7} | {'n':>4}"
        )
        print(header)
        print("-" * len(header))

        base = None
        for strat in chosen:
            n, dca, dcc = rank_curve(
                df, raw_dir, k, strat,
                None if site_type_filter < 0 else site_type_filter,
                max_center_dist,
            )
            dca0, dcat = dca[0], dca[report_rank]
            dcc0, dcct = dcc[0], dcc[report_rank]
            row = (
                f"{strat:<28} "
                f"{dca0:6.3f} {dcat:6.3f} {dcat - dca0:7.3f} | "
                f"{dcc0:6.3f} {dcct:6.3f} {dcct - dcc0:7.3f} | {n:4d}"
            )
            if strat == "confidence_0":
                base = (dca0, dcc0)
                row += "   <- baseline (reproduction check)"
            elif base is not None:
                row += f"   dDCA0={dca0 - base[0]:+.3f} dDCC0={dcc0 - base[1]:+.3f}"
            print(row)


if __name__ == "__main__":
    main()

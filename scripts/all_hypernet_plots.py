import argparse
from collections import defaultdict
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from coolchic.eval.bd_rate import bd_rates_summary_anchor_name
from coolchic.eval.hypernet import (
    get_hypernet_flops,
    plot_hypernet_rd,
    plot_hypernet_rd_avg,
)
from coolchic.eval.results import SummaryEncodingMetrics
from coolchic.hypernet.hypernet import (
    CoolchicWholeNet,
    DeltaWholeNet,
    NOWholeNet,
    SmallDeltaWholeNet,
)
from coolchic.utils.paths import ANCHOR_NAME, DATA_DIR, DATASET_NAME


def parse_hypernet_metrics(
    sweep_path: Path, dataset: DATASET_NAME, premature: bool = False
) -> dict[str, list[SummaryEncodingMetrics]]:
    """Metrics saved by hypernet training jobs are csv files
    with the following columns: seq_name, rate_bpp, psnr_db, mse.
    Maybe lmbda.

    We want to parse the files and get them into SummaryEncodingMetrics,
    those are the ones used by bd rate scripts.
    """
    runs = [
        run
        for run in sweep_path.iterdir()
        if run.is_dir() and run.name.startswith("config_")
    ]
    all_metrics: dict[str, list[SummaryEncodingMetrics]] = defaultdict(list)
    lmbdas = {"00": 0.0001, "01": 0.0004, "02": 0.001, "03": 0.004, "04": 0.02}

    for run in runs:
        run_lmbda = lmbdas[run.stem.split("_")[-1]]
        results_path = (
            run / "premature_eval" if premature else run
        ) / f"{dataset}_results.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        results = pd.read_csv(results_path)
        for row in results.itertuples():
            row = cast(SummaryEncodingMetrics, row)  # Hey pyright, trust me bro.
            all_metrics[row.seq_name].append(
                SummaryEncodingMetrics(
                    seq_name=row.seq_name,
                    rate_bpp=row.rate_bpp,
                    psnr_db=row.psnr_db,
                    lmbda=row.lmbda if "lmbda" in row else run_lmbda,  # pyright: ignore
                )
            )

    return all_metrics


def print_bd(
    metrics: dict[str, list[SummaryEncodingMetrics]],
    anchor_name: ANCHOR_NAME,
    dataset: DATASET_NAME,
):
    print(f"Results for anchor {anchor_name}:")
    bd_rates = bd_rates_summary_anchor_name(metrics, anchor_name, dataset)
    for seq, r in bd_rates.items():
        print(f"{seq}: {r}")
    print(f"Average: {sum(bd_rates.values()) / len(bd_rates)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_path", type=Path, required=True)
    parser.add_argument("--premature", action="store_true")
    parser.add_argument("--wholenet_cls", type=str, default="NOWholeNet")
    parser.add_argument("--compare_no_path", type=Path, default=None)
    parser.add_argument("--compare_premature", action="store_true", default=False)
    parser.add_argument(
        "--dataset", type=str, choices=["kodak", "clic20-pro-valid"], required=True
    )
    args = parser.parse_args()
    if not args.sweep_path.exists():
        raise FileNotFoundError(f"Path not found: {args.sweep_path}")

    sweep_path = args.sweep_path
    metrics = parse_hypernet_metrics(
        sweep_path, dataset=args.dataset, premature=args.premature
    )
    # BD rates for coolchic, hm, jpeg
    for anchor in ["coolchic", "hm", "jpeg"]:
        print_bd(metrics, anchor, args.dataset)  # pyright: ignore

    # BD rate vs computational cost
    avg_bd = sum(
        (bd_rates := bd_rates_summary_anchor_name(metrics, "hm", args.dataset).values())
    ) / len(bd_rates)

    wholenet_cls_dict = {
        "NOWholeNet": NOWholeNet,
        "DeltaWholeNet": DeltaWholeNet,
        "CoolchicWholeNet": CoolchicWholeNet,
        "SmallDeltaWholeNet": SmallDeltaWholeNet,
    }
    try:
        wholenet_cls = wholenet_cls_dict[args.wholenet_cls]
    except KeyError:
        raise ValueError(
            f"Invalid wholenet_cls: {args.wholenet_cls}. "
            f"Valid options are: {list(wholenet_cls_dict.keys())}"
        )

    comp_cost = get_hypernet_flops(wholenet_cls)
    print(f"{avg_bd=}, {comp_cost=:.3e}")

    ###### RD PLOTS ######
    # Compare with NO coolchic, if provided.
    if args.compare_no_path:
        no_metrics = parse_hypernet_metrics(
            args.compare_no_path, dataset=args.dataset, premature=args.compare_premature
        )
        no_df = pd.DataFrame(
            [s.model_dump() for seq_res in no_metrics.values() for s in seq_res]
        )
        no_df["anchor"] = "NOCoolChic"
    else:
        no_df = pd.DataFrame()

    df = pd.DataFrame([s.model_dump() for seq_res in metrics.values() for s in seq_res])
    df["anchor"] = "hnet"
    df = pd.concat([df, no_df])
    df = df.sort_values(by=["seq_name", "lmbda"])  # So plot comes out nice.

    for img in (DATA_DIR / args.dataset).glob("*.png"):
        seq_name = img.stem
        plot_hypernet_rd(seq_name, df, dataset=args.dataset)

    plot_hypernet_rd_avg(df, dataset=args.dataset)

    plt.show()

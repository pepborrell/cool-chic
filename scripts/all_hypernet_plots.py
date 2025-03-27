import argparse
from collections import defaultdict
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from coolchic.eval.bd_rate import bd_rates_summary_anchor_name
from coolchic.eval.hypernet import get_hypernet_flops, plot_hypernet_rd
from coolchic.eval.results import SummaryEncodingMetrics
from coolchic.hypernet.hypernet import (
    CoolchicWholeNet,
    DeltaWholeNet,
    NOWholeNet,
)
from coolchic.utils.paths import ANCHOR_NAMES


def parse_hypernet_metrics(sweep_path: Path, premature: bool = False):
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
        ) / "kodak_results.csv"
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
    metrics: dict[str, list[SummaryEncodingMetrics]], anchor_name: ANCHOR_NAMES
):
    print(f"Results for anchor {anchor_name}:")
    bd_rates = bd_rates_summary_anchor_name(metrics, anchor_name)
    for seq, r in bd_rates.items():
        print(f"{seq}: {r}")
    print(f"Average: {sum(bd_rates.values()) / len(bd_rates)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_path", type=Path, required=True)
    parser.add_argument("--premature", action="store_true")
    parser.add_argument("--wholenet_cls", type=str, default="NOWholeNet")
    args = parser.parse_args()
    if not args.sweep_path.exists():
        raise FileNotFoundError(f"Path not found: {args.sweep_path}")

    sweep_path = args.sweep_path
    metrics = parse_hypernet_metrics(sweep_path, args.premature)
    # BD rates for coolchic, hm, jpeg
    print_bd(metrics, "coolchic")
    print_bd(metrics, "hm")
    print_bd(metrics, "jpeg")

    # BD rate vs computational cost
    avg_bd = sum(
        (bd_rates := bd_rates_summary_anchor_name(metrics, "hm").values())
    ) / len(bd_rates)
    wholenet_cls = {
        "NOWholeNet": NOWholeNet,
        "DeltaWholeNet": DeltaWholeNet,
        "CoolchicWholeNet": CoolchicWholeNet,
    }[args.wholenet_cls]
    comp_cost = get_hypernet_flops(wholenet_cls)
    print(f"{avg_bd=}, {comp_cost=:.3e}")

    # RD plots
    for i in range(1, 25):
        seq_name = f"kodim{i:02d}"
        df = pd.DataFrame(
            [s.model_dump() for seq_res in metrics.values() for s in seq_res]
        )
        df["anchor"] = "hnet"
        df = df.sort_values(by=["seq_name", "lmbda"])  # So plot comes out nice.
        plot_hypernet_rd(seq_name, df)
    plt.show()

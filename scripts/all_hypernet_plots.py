import argparse
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
import pandas as pd

from coolchic.eval.bd_rate import bd_rates_summary_anchor_name
from coolchic.eval.hypernet import (
    get_hypernet_flops,
    plot_hypernet_rd,
    plot_hypernet_rd_avg,
)
from coolchic.eval.results import SummaryEncodingMetrics, parse_hypernet_metrics
from coolchic.hypernet.hypernet import (
    CoolchicWholeNet,
    DeltaWholeNet,
    NOWholeNet,
    SmallDeltaWholeNet,
)
from coolchic.utils.paths import ANCHOR_NAME, DATA_DIR, DATASET_NAME


def print_bd(
    metrics: dict[str, list[SummaryEncodingMetrics]],
    anchor_name: ANCHOR_NAME,
    dataset: DATASET_NAME,
    only_latent_rate: bool,
):
    print(f"Results for anchor {anchor_name}:")
    bd_rates = bd_rates_summary_anchor_name(
        metrics, anchor_name, dataset, only_latent_rate=only_latent_rate
    )
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
    parser.add_argument(
        "--only_latent_rate",
        action="store_true",
        help="If set, will only use latent rate in BD rate computation.",
    )
    args = parser.parse_args()
    if not args.sweep_path.exists():
        raise FileNotFoundError(f"Path not found: {args.sweep_path}")

    sweep_path = args.sweep_path
    metrics = parse_hypernet_metrics(
        sweep_path, dataset=args.dataset, premature=args.premature
    )
    # BD rates for coolchic, hm, jpeg
    for anchor in get_args(ANCHOR_NAME):
        print_bd(metrics, anchor, args.dataset, only_latent_rate=args.only_latent_rate)

    # BD rate vs computational cost
    avg_bd = sum(
        (
            bd_rates := bd_rates_summary_anchor_name(
                metrics, "hm", args.dataset, only_latent_rate=args.only_latent_rate
            ).values()
        )
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

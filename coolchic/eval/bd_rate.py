"""
Compares the BD rate of two runs, specified by the path to its results.
"""

import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Any

import seaborn as sns
import yaml
from pydantic import BaseModel

from ..utils.bjontegaard_metric import BD_RATE


class EncodingMetrics(BaseModel):
    loss: float
    nn_bpp: float
    latent_bpp: float
    psnr_db: float
    total_rate_bpp: float
    lmbda: float
    feature_rate_bpp_00: float
    feature_rate_bpp_01: float
    feature_rate_bpp_02: float
    feature_rate_bpp_03: float
    feature_rate_bpp_04: float
    feature_rate_bpp_05: float
    feature_rate_bpp_06: float
    arm_rate_bpp: float
    upsampling_rate_bpp: float
    synthesis_rate_bpp: float


def split_row(row: str) -> list[Any]:
    split_char = " "
    if "\t" in row:
        split_char = "\t"
    return [elem.strip() for elem in row.split(split_char) if elem != ""]


def parse_result_metrics(results_dir: Path) -> EncodingMetrics | None:
    # Path to results looks like: results/exps/2024-11-15/kodim01/
    best_results_file = results_dir / "frame_000results_best.tsv"
    if not best_results_file.exists():
        print(f"Warning: file {best_results_file} was not found.")
        return
    with open(best_results_file, "r") as f:
        raw_metrics = f.read().strip()
    metric_names, metric_values = [split_row(row) for row in raw_metrics.split("\n")]
    assert len(metric_names) == len(metric_values)
    metrics = EncodingMetrics(
        **{
            n: float(v) if v.isnumeric() else v
            for n, v in zip(metric_names, metric_values)
        }
    )
    return metrics


def get_run_config(results_dir: Path) -> dict[str, Any]:
    config_file = results_dir / "param.txt"
    with open(config_file, "r") as f:
        config = yaml.unsafe_load(f)
    parameter_names = ["lmbda", "input"]
    params = {p: config[p] for p in parameter_names}
    assert isinstance(params["input"], Path)  # for linter to understand
    params["seq_name"] = params["input"].stem
    return params


class SummaryEncodingMetrics(BaseModel):
    seq_name: str
    lmbda: float
    rate_bpp: float
    psnr_db: float


def parse_result_summary(summary_file: Path) -> dict[str, list[SummaryEncodingMetrics]]:
    with open(summary_file, "r") as f:
        metric_names = split_row(f.readline().strip())
        raw_metrics = f.readlines()
    results = defaultdict(list)
    for line in raw_metrics:
        line_metrics = {n: v for n, v in zip(metric_names, split_row(line))}
        results[line_metrics["seq_name"]].append(SummaryEncodingMetrics(**line_metrics))
    return results


def gen_run_summary(run_dir: Path) -> SummaryEncodingMetrics | None:
    metrics = parse_result_metrics(run_dir)
    if metrics is None:
        return
    params = get_run_config(run_dir)
    all_data = metrics.model_dump() | params
    all_data["rate_bpp"] = all_data["total_rate_bpp"]
    return SummaryEncodingMetrics(**all_data)


def full_run_summary(run_suite_dir: Path) -> dict[str, list[SummaryEncodingMetrics]]:
    summaries = defaultdict(list)
    for kodim_config in run_suite_dir.iterdir():
        all_runs = [subdir for subdir in kodim_config.iterdir() if subdir.is_dir()]
        for dir in all_runs:
            summary = gen_run_summary(dir)
            if summary is not None:
                summaries[summary.seq_name].append(summary)
    return summaries


def bd_rate_summaries(
    ref_sum: list[SummaryEncodingMetrics], other_sum: list[SummaryEncodingMetrics]
) -> float:
    sorted_ref = sorted(ref_sum, key=lambda s: s.lmbda)
    sorted_other = sorted(other_sum, key=lambda s: s.lmbda)

    def extract_rate_distortion(
        summaries: list[SummaryEncodingMetrics],
    ) -> tuple[list[float], list[float]]:
        return [s.rate_bpp for s in summaries], [s.psnr_db for s in summaries]

    return BD_RATE(
        *extract_rate_distortion(sorted_ref), *extract_rate_distortion(sorted_other)
    )


def gen_rd_plots(
    summaries: list[SummaryEncodingMetrics],
    other_sums: list[SummaryEncodingMetrics] | None = None,
) -> None:
    df = pd.DataFrame([s.model_dump() for s in summaries])
    df["run"] = "reference"
    if other_sums:
        other_df = pd.DataFrame([s.model_dump() for s in other_sums])
        other_df["run"] = "other"
        other_df.seq_name = other_df.seq_name.apply(lambda s: s + "_other")
        df.seq_name = df.seq_name.apply(lambda s: s + "_ref")
        df = pd.concat([df, other_df])
    sns.lineplot(df, x="rate_bpp", y="psnr_db", hue="seq_name", marker="o")
    plt.show()


def print_md_table(results: dict[str, float]) -> None:
    output = "| seq_name | bd rate |\n"
    output += "| :------- | ------: |\n"
    for seq, value in results.items():
        output += f"| {seq} | {value:.2f} |\n"
    print(output)


if __name__ == "__main__":
    runs_path = Path("results/exps/copied/2024-11-26/")
    run_summaries = full_run_summary(runs_path)
    og_summary_dir = Path("results/image/kodak/results.tsv")
    og_summary = parse_result_summary(og_summary_dir)

    results = {}
    for seq_name in og_summary:
        bd_rate = bd_rate_summaries(og_summary[seq_name], run_summaries[seq_name])
        results[seq_name] = bd_rate
    print_md_table(results)

    # gen_rd_plots([sum for sums in og_summary.values() for sum in sums])
    some_images = [f"kodim{num:02}" for num in range(1, 9)]
    gen_rd_plots(
        [sum for seq_name in some_images for sum in og_summary[seq_name]],
        [sum for seq_name in some_images for sum in run_summaries[seq_name]],
    )

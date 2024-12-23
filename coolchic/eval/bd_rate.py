"""
Compares the BD rate of two runs, specified by the path to its results.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
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
    parameter_names = ["lmbda", "input", "enc_cfg"]
    params = {p: config[p] for p in parameter_names}
    assert isinstance(params["input"], Path)  # for linter to understand
    params["seq_name"] = params["input"].stem
    # Bubbling up encoding parameters so we can use them later.
    params["n_itr"] = params["enc_cfg"]["n_itr"]
    params["n_train_loops"] = params["enc_cfg"]["n_train_loops"]
    return params


class SummaryEncodingMetrics(BaseModel):
    seq_name: str
    lmbda: float
    rate_bpp: float
    psnr_db: float
    # Fields not in the cool-chic summaries but we use them in our analyses.
    n_itr: int | None = None
    n_train_loops: int | None = None


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
    # Renaming for consistency.
    all_data["rate_bpp"] = all_data["total_rate_bpp"]
    if all_data["n_itr"] is None:
        all_data["n_itr"] = all_data["enc_cfg"]["recipe"]["all_phases"][0]["max_itr"]
    return SummaryEncodingMetrics(**all_data)


def read_real_rate(dir_path: Path) -> float:
    with open(dir_path / "real_rate_bpp.txt", "r") as f:
        return float(f.read().strip())


def full_run_summary(
    run_suite_dir: Path, real_rate: bool = False
) -> dict[str, list[SummaryEncodingMetrics]]:
    summaries = defaultdict(list)
    # We get all files we will need to review.
    # We do this because we allow to give an arbitrary path and we analyse all runs under it.
    all_run_results = [
        file for file in run_suite_dir.rglob("*results_best.tsv") if file.is_file()
    ]
    for run in all_run_results:
        dir = run.parent
        summary = gen_run_summary(dir)
        if summary is not None:
            if real_rate:
                summary.rate_bpp = read_real_rate(dir)
            summaries[summary.seq_name].append(summary)
    return summaries


def bd_rate_summaries(
    ref_sum: list[SummaryEncodingMetrics], other_sum: list[SummaryEncodingMetrics]
) -> float:
    # THE REFERENCE IS THE ANCHOR!
    sorted_ref = sorted(ref_sum, key=lambda s: s.lmbda)
    sorted_other = sorted(other_sum, key=lambda s: s.lmbda)

    def extract_rate_distortion(
        summaries: list[SummaryEncodingMetrics],
    ) -> tuple[list[float], list[float]]:
        return [s.rate_bpp for s in summaries], [s.psnr_db for s in summaries]

    return BD_RATE(
        *extract_rate_distortion(sorted_ref), *extract_rate_distortion(sorted_other)
    )


def avg_bd_rate_summary_paths(summary_path: Path, anchor_path: Path) -> float:
    # checking that paths are as expected.
    assert summary_path.exists() and summary_path.is_file()
    assert anchor_path.exists() and anchor_path.is_file()

    summary = parse_result_summary(summary_path)
    a_summary = parse_result_summary(anchor_path)
    results = []
    for seq_name in summary:
        # REMEMBER: the anchor goes first.
        bd_rate = bd_rate_summaries(a_summary[seq_name], summary[seq_name])
        results.append(bd_rate)
    return np.mean(results)


def bd_rates_from_paths(
    runs_path: Path, anchor_path: Path, real_rate: bool = False
) -> list[float]:
    # checking that paths are as expected.
    assert runs_path.is_dir()
    assert anchor_path.exists() and anchor_path.is_file()

    run_summaries = full_run_summary(runs_path, real_rate)
    og_summary = parse_result_summary(anchor_path)
    results = []
    for seq_name in og_summary:
        bd_rate = bd_rate_summaries(og_summary[seq_name], run_summaries[seq_name])
        results.append(bd_rate)
    return results

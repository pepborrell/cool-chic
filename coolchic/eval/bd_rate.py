"""
Compares the BD rate of two runs, specified by the path to its results.
"""

from collections import defaultdict
from typing import Any
from pathlib import Path

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


def parse_result_metrics(results_dir: Path) -> EncodingMetrics:
    # Path to results looks like: results/exps/2024-11-15/kodim01/
    best_results_file = results_dir / "frame_000results_best.tsv"
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


class SummaryEncodingMetrics(BaseModel):
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


def bd_rate_results(results_dir1: Path, results_dir2: Path) -> float:
    # TODO: get results for several lambdas.
    metrics1 = parse_result_metrics(results_dir1)
    metrics2 = parse_result_metrics(results_dir2)
    return BD_RATE(
        metrics1.total_rate_bpp,
        metrics1.psnr_db,
        metrics2.total_rate_bpp,
        metrics2.psnr_db,
    )


def bd_rate_with_ref(results_dir: Path, ref_dir: Path, img_name: str) -> float:
    # TODO: get results with several lambdas.
    metrics = parse_result_metrics(results_dir)
    refs = parse_result_summary(ref_dir)

    ref = refs[img_name]
    ref_rates = [this_ref.rate_bpp for this_ref in ref]
    ref_psnrs = [this_ref.psnr_db for this_ref in ref]
    return BD_RATE(metrics.total_rate_bpp, metrics.psnr_db, ref_rates, ref_psnrs)


if __name__ == "__main__":
    for num in range(1, 25):
        img_name = f"kodim{num:02}"
        p = Path(f"results/exps/copied/{img_name}/hop/")
        ref = Path("results/image/kodak/results.tsv")
        print(f"{img_name=}, {bd_rate_with_ref(p, ref, img_name)=}")

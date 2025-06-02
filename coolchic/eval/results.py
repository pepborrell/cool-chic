from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from pydantic import BaseModel

from coolchic.enc.training.test import FrameEncoderLogs
from coolchic.utils.paths import DATASET_NAME


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
    rate_latent_bpp: float | None = None
    rate_nn_bpp: float | None = None
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
    return dict(results)


def result_summary_to_df(summary_path: Path) -> pd.DataFrame:
    summaries = parse_result_summary(summary_path)
    all_data = []
    for seq_name in summaries:
        all_data.extend([s.model_dump() for s in summaries[seq_name]])
    return pd.DataFrame(all_data)


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


def log_to_results(logs: FrameEncoderLogs, seq_name: str) -> SummaryEncodingMetrics:
    assert logs.total_rate_bpp is not None  # To make pyright happy.
    assert logs.psnr_db is not None  # To make pyright happy.
    return SummaryEncodingMetrics(
        seq_name=seq_name,
        rate_bpp=logs.total_rate_bpp,
        psnr_db=logs.psnr_db,
        lmbda=logs.encoding_iterations_cnt,
    )


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
                    rate_latent_bpp=row.rate_latent_bpp,
                    rate_nn_bpp=row.rate_nn_bpp,
                    psnr_db=row.psnr_db,
                    lmbda=row.lmbda if "lmbda" in row else run_lmbda,  # pyright: ignore
                )
            )

    return all_metrics

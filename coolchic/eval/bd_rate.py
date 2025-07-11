"""
Compares the BD rate of two runs, specified by the path to its results.
"""

from pathlib import Path

import numpy as np

from coolchic.eval.results import (
    SummaryEncodingMetrics,
    full_run_summary,
    parse_result_summary,
)
from coolchic.utils.bjontegaard_metric import BD_RATE
from coolchic.utils.paths import ALL_ANCHORS, ANCHOR_NAME, DATASET_NAME


def bd_rate_summaries(
    ref_sum: list[SummaryEncodingMetrics],
    other_sum: list[SummaryEncodingMetrics],
    only_latent_rate: bool,
) -> float:
    # THE REFERENCE IS THE ANCHOR!
    sorted_ref = sorted(ref_sum, key=lambda s: s.lmbda)
    sorted_other = sorted(other_sum, key=lambda s: s.lmbda)

    def extract_rate_distortion(
        summaries: list[SummaryEncodingMetrics], only_latent_rate: bool = False
    ) -> tuple[list[float], list[float]]:
        if only_latent_rate:
            assert all(s.rate_latent_bpp is not None for s in summaries)
        return [
            s.rate_bpp if not only_latent_rate else s.rate_latent_bpp
            for s in summaries  # pyright: ignore
        ], [s.psnr_db for s in summaries]

    return BD_RATE(
        *extract_rate_distortion(sorted_ref),
        *extract_rate_distortion(sorted_other, only_latent_rate),
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
        bd_rate = bd_rate_summaries(
            a_summary[seq_name], summary[seq_name], only_latent_rate=False
        )
        results.append(bd_rate)
    return np.mean(results).item()


def bd_rates_summary_anchor_name(
    summary: dict[str, list[SummaryEncodingMetrics]],
    anchor: ANCHOR_NAME,
    dataset: DATASET_NAME,
    only_latent_rate: bool,
) -> dict[str, float]:
    a_summary = parse_result_summary(ALL_ANCHORS[dataset][anchor])
    results: dict[str, float] = {}
    for seq_name in summary:
        # REMEMBER: the anchor goes first.
        if seq_name not in a_summary:
            print(
                f"Warning: {seq_name} not found in anchor {anchor}. Skipping BD rate calculation."
            )
            continue
        results[seq_name] = bd_rate_summaries(
            a_summary[seq_name], summary[seq_name], only_latent_rate=only_latent_rate
        )
    return results


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
        bd_rate = bd_rate_summaries(
            og_summary[seq_name], run_summaries[seq_name], only_latent_rate=False
        )
        results.append(bd_rate)
    return results

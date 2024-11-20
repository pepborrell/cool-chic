"""
Compares the BD rate of two runs, specified by the path to its results.
"""

from pathlib import Path

from pydantic import BaseModel

from coolchic.utils.bjontegaard_metric import BD_RATE


class EncodingMetrics(BaseModel):
    loss: float
    nn_bpp: float
    latent_bpp: float
    psnr_db: float
    total_rate_bpp: float
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
    # UNWANTED METRICS
    # arm_weight_q_step
    # arm_bias_q_step
    # upsampling_weight_q_step
    # upsampling_bias_q_step
    # synthesis_weight_q_step
    # synthesis_bias_q_step
    # arm_weight_exp_cnt
    # arm_bias_exp_cnt
    # upsampling_weight_exp_cnt
    # upsampling_bias_exp_cnt
    # synthesis_weight_exp_cnt
    # synthesis_bias_exp_cnt
    # lmbda
    # time_sec
    # itr
    # mac_decoded_pixel
    # img_size
    # n_pixels
    # display_order
    # coding_order
    # seq_name


def parse_result_metrics(results_dir: Path) -> EncodingMetrics:
    # Path to results looks like: results/exps/2024-11-15/kodim01/
    best_results_file = results_dir / "frame_000results_best.tsv"
    with open(best_results_file, "r") as f:
        raw_metrics = f.read()
    metric_names, metric_values = raw_metrics.split("\n")
    metrics = EncodingMetrics(
        **{
            n: float(v)
            for n, v in zip(metric_names.split("\t"), metric_values.split("\t"))
        }
    )
    return metrics


def bd_rate_results(results_dir1: Path, results_dir2: Path) -> float:
    metrics1 = parse_result_metrics(results_dir1)
    metrics2 = parse_result_metrics(results_dir2)
    return BD_RATE(metrics1.nn_bpp, metrics1.psnr_db, metrics2.nn_bpp, metrics2.psnr_db)

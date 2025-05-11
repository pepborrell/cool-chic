from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from coolchic.enc.io.io import load_frame_data_from_file
from coolchic.enc.training.presets import PresetC3x
from coolchic.enc.training.test import FrameEncoderLogs, test
from coolchic.eval.results import result_summary_to_df
from coolchic.hypernet.hypernet import CoolchicHyperNet, NOWholeNet, WholeNet
from coolchic.utils.coolchic_types import get_coolchic_structs
from coolchic.utils.paths import ALL_ANCHORS
from coolchic.utils.types import DecoderConfig, HyperNetConfig, PresetConfig


def compare_kodak_res(results: pd.DataFrame) -> pd.DataFrame:
    res_sums = []
    for anchor, results_path in ALL_ANCHORS.items():
        df = result_summary_to_df(results_path)
        df["anchor"] = anchor
        res_sums.append(df)

    assert "anchor" in results.columns, "Anchor column not found in results"
    res_sums.append(results)
    all_df = pd.concat(res_sums)

    return all_df


def plot_hypernet_rd(kodim_name: str, results: pd.DataFrame):
    all_df = compare_kodak_res(results)
    all_df = all_df.loc[all_df["seq_name"] == kodim_name]

    fig, ax = plt.subplots()
    sns.lineplot(
        all_df,
        x="rate_bpp",
        y="psnr_db",
        hue="anchor",
        marker="o",
        markeredgecolor="none",
        ax=ax,
        sort=False,
    )
    ax.set_title(f"RD curve for {kodim_name}")
    return fig, ax


def plot_hypernet_rd_avg(results: pd.DataFrame):
    """Plots the average RD plot for the whole dataset in results."""
    all_df = compare_kodak_res(results)
    mean_df = (
        all_df.groupby(["anchor", "lmbda"])
        .agg({"rate_bpp": "mean", "psnr_db": "mean"})
        .reset_index()
    )
    fig, ax = plt.subplots()
    sns.lineplot(
        mean_df,
        x="rate_bpp",
        y="psnr_db",
        hue="anchor",
        marker="o",
        markeredgecolor="none",
        ax=ax,
        sort=False,
    )
    ax.set_title("Average RD curve for all kodim images")
    return fig, ax


def is_above_anchor_curve(
    point: tuple[float, float], anchor_curve: list[tuple[float, float]]
):
    # Assuming the anchor curve is increasing.
    for anchor_point in anchor_curve:
        if point[0] < anchor_point[0] and point[1] > anchor_point[1]:
            return True
    return False


def find_crossing_it(
    kodim_name: str, results: pd.DataFrame, run_name: str, anchor_name: str
) -> int:
    all_df = compare_kodak_res(results)
    all_df = all_df.loc[all_df["seq_name"] == kodim_name]

    def get_curve_for_anchor(anchor_name: str, sort: bool = False):
        anchor_df = all_df.loc[all_df["anchor"] == anchor_name]
        curve = list(zip(anchor_df["rate_bpp"], anchor_df["psnr_db"]))
        if sort:
            return sorted(curve, key=lambda x: x[0])
        return curve

    anchor_curve = get_curve_for_anchor(anchor_name, sort=True)
    run_points = get_curve_for_anchor(run_name)

    for i, point in enumerate(run_points):
        if is_above_anchor_curve(point, anchor_curve):
            return i
    return -1


def coolchic_test_hypernet(
    model: WholeNet, img_path: Path, lmbda: float
) -> FrameEncoderLogs:
    # Load image.
    frame_data = load_frame_data_from_file(str(img_path), 0)
    img = frame_data.data
    assert isinstance(img, torch.Tensor)  # To make pyright happy.
    device = next(model.parameters()).device
    img = img.to(device)

    cc_encoder = model.image_to_coolchic(img, stop_grads=True)

    # Placeholder preset config.
    # I don't think it's being used for anything useful.
    # Based on C3x.
    preset_config = PresetConfig(
        preset_name=(tmp_preset := PresetC3x()).preset_name,
        warmup=tmp_preset.warmup,
        all_phases=tmp_preset.all_phases,
    )

    frame, frame_encoder_manager, frame_enc = get_coolchic_structs(
        lmbda=lmbda,
        preset_config=preset_config,
        dec_cfg=model.config.dec_cfg,
        cc_encoder=cc_encoder,
        frame_data=frame_data,
    )
    logs = test(frame_enc, frame, frame_encoder_manager)
    return logs


def get_hypernet_flops(wholenet_cls: type[WholeNet]) -> int:
    model = wholenet_cls(config=HyperNetConfig(dec_cfg=DecoderConfig()))
    if not hasattr(model, "hypernet"):
        # Dealing with NOWholeNet.
        assert isinstance(model, NOWholeNet)  # For pyright to understand.
        total_flops = model.encoder.get_flops()
        return total_flops
    # We either have a WholeNet or a DeltaWholeNet.
    hnet = model.hypernet
    assert isinstance(hnet, CoolchicHyperNet)  # For pyright to understand.

    return hnet.get_flops()

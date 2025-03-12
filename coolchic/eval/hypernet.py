import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from coolchic.eval.results import result_summary_to_df
from coolchic.utils.paths import ALL_ANCHORS


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

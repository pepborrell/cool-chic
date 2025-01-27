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

    results["anchor"] = "hypernet"
    res_sums.append(results)
    all_df = pd.concat(res_sums)

    return all_df


def plot_hypernet_rd(kodim_name: str, results: pd.DataFrame):
    all_df = compare_kodak_res(results)
    all_df = all_df.loc[all_df["seq_name"] == kodim_name]

    fig, ax = plt.subplots()
    sns.lineplot(all_df, x="rate_bpp", y="psnr_db", hue="anchor", marker="o", ax=ax)
    ax.set_title(f"RD curve for {kodim_name}")
    return fig, ax

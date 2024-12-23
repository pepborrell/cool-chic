import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from coolchic.eval.bd_rate import (
    SummaryEncodingMetrics,
)

sns.set_theme(context="notebook", style="whitegrid")


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
    sns.despine()


def print_md_table(results: dict[str, float]) -> None:
    output = "| seq_name | bd rate |\n"
    output += "| :------- | ------: |\n"
    for seq, value in results.items():
        output += f"| {seq} | {value:.2f} |\n"
    print(output)


def plot_bd_rate_n_itr(
    df: pd.DataFrame, anchor_name: str | None = None, bd_vs_cc: float | None = None
):
    """df expected to have these columns:
    * avg_bd_rate
    * n_itr
    * n_train_loops
    """
    fig, ax = plt.subplots()
    sns.lineplot(
        df, x="n_itr", y="avg_bd_rate", hue="n_train_loops", ax=ax, marker="o"
    ).set_title("BD-rate vs number of iterations per loop" + f" anchor={anchor_name}")
    if (df.avg_bd_rate >= 0).all():
        ax.set_ylim(0, None)
    if bd_vs_cc:
        ax.axhline(y=bd_vs_cc, color="red", linestyle="--", linewidth=2)
    sns.despine(ax=ax)
    return fig


def plot_bd_rate_total_itr(
    df: pd.DataFrame,
    anchor_name: str | None = None,
    bd_vs_cc: float | None = None,
    real_rate: bool = False,
):
    """df expected to have these columns:
    * avg_bd_rate
    * n_itr
    * n_train_loops
    """
    df = df.assign(total_n_itr=df.n_itr * df.n_train_loops)
    fig, ax = plt.subplots()
    sns.lineplot(
        df, x="total_n_itr", y="avg_bd_rate", hue="n_train_loops", ax=ax, marker="o"
    ).set_title(
        "BD-rate vs number of real iterations"
        + f" anchor={anchor_name}"
        + f" real_rate={real_rate}"
    )
    if (df.avg_bd_rate >= 0).all():
        ax.set_ylim(0, None)
    if bd_vs_cc:
        ax.axhline(y=bd_vs_cc, color="red", linestyle="--", linewidth=2)
    sns.despine(ax=ax)
    return fig


def gen_rd_iterations(anchors: pd.DataFrame, summaries: pd.DataFrame):
    anchors.n_itr = "anchor"

    fig, ax = plt.subplots()
    sns.lineplot(
        data=anchors,
        x="rate_bpp",
        y="psnr_db",
        marker="o",
        hue="n_itr",
        ax=ax,
    )
    sns.scatterplot(
        data=summaries, x="rate_bpp", y="psnr_db", hue="n_itr", ax=ax, marker="o"
    )
    sns.despine()
    plt.title(f"Rate-Distortion curve. Image={summaries.seq_name.iloc[0]}")
    return fig, ax

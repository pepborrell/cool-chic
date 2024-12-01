import seaborn as sns
import matplotlib.pyplot as plt
from coolchic.eval.bd_rate import avg_bd_rate_from_paths, full_run_summary
from pathlib import Path
import pandas as pd


def plot_bd_rate_n_itr(df: pd.DataFrame):
    """df expected to have these columns:
    * avg_bd_rate
    * n_itr
    * n_train_loops
    """
    df = df.assign(total_n_itr=df.n_itr * df.n_train_loops)
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(df, x="n_itr", y="avg_bd_rate", hue="n_itr", ax=ax)
    return fig


def plot_test():
    all_runs = Path("results/exps/n_it-grid/")
    og_summary_dir = Path("results/image/kodak/results.tsv")
    all_data = []
    for run_path in all_runs.iterdir():
        run_summaries = full_run_summary(run_path)
        n_itr = list(run_summaries.values())[0][0].n_itr
        n_train_loops = list(run_summaries.values())[0][0].n_train_loops
        # All runs in this dir have the same n_itr and n_train_loops, but let's make sure.
        assert all(
            config.n_itr == n_itr and config.n_train_loops == n_train_loops
            for img_configs in run_summaries.values()
            for config in img_configs
        )
        all_data.append(
            {
                "avg_bd": avg_bd_rate_from_paths(run_path, og_summary_dir),
                "n_itr": n_itr,
                "n_train_loops": n_train_loops,
            }
        )
    df = pd.DataFrame(all_data)
    fig = plot_bd_rate_n_itr(df)
    fig.show()

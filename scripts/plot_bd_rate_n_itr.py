from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from coolchic.eval.bd_rate import (
    avg_bd_rate_summary_paths,
    bd_rates_from_paths,
    full_run_summary,
)
from coolchic.eval.plotting import plot_bd_rate_n_itr, plot_bd_rate_total_itr

all_runs = Path("results/exps/copied/n_it-grid/")
og_summary_dir = Path("results/image/kodak/results.tsv")
hm_summary_dir = Path("results/image/kodak/hm.tsv")
jpeg_summary_dir = Path("results/image/kodak/jpeg.tsv")

all_anchors = {
    "cool-chic": og_summary_dir,
    "hm": hm_summary_dir,
    "jpeg": jpeg_summary_dir,
}
all_data = []
for run_path in tqdm(list(all_runs.iterdir())):
    run_summaries = full_run_summary(run_path)
    n_itr = list(run_summaries.values())[0][0].n_itr
    n_train_loops = list(run_summaries.values())[0][0].n_train_loops
    # All runs in this dir have the same n_itr and n_train_loops, but let's make sure.
    assert all(
        config.n_itr == n_itr and config.n_train_loops == n_train_loops
        for img_configs in run_summaries.values()
        for config in img_configs
    )
    for anchor in all_anchors:
        all_data.extend(
            {
                "avg_bd_rate": bd_rate,
                "n_itr": n_itr,
                "n_train_loops": n_train_loops,
                "anchor": anchor,
            }
            for bd_rate in bd_rates_from_paths(run_path, all_anchors[anchor])
        )

bd_vs_best_cc = {
    anchor: avg_bd_rate_summary_paths(og_summary_dir, all_anchors[anchor])
    if anchor != "cool-chic"
    else None
    for anchor in all_anchors
}

plots = []
df = pd.DataFrame(all_data)
for anchor in all_anchors:
    a_df = df.loc[df.anchor == anchor]
    f1 = plot_bd_rate_n_itr(a_df, anchor_name=anchor, bd_vs_cc=bd_vs_best_cc[anchor])
    f2 = plot_bd_rate_total_itr(
        a_df, anchor_name=anchor, bd_vs_cc=bd_vs_best_cc[anchor]
    )
    plots.extend([f1, f2])
plt.show()

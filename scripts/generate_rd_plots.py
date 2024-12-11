from pathlib import Path

import matplotlib.pyplot as plt

from coolchic.eval.bd_rate import (
    bd_rate_summaries,
    full_run_summary,
    parse_result_summary,
)
from coolchic.eval.plotting import gen_rd_plots, print_md_table

if __name__ == "__main__":
    runs_path = Path("results/exps/copied/2024-11-26/")
    run_summaries = full_run_summary(runs_path)
    og_summary_dir = Path("results/image/kodak/results.tsv")
    og_summary = parse_result_summary(og_summary_dir)

    results = {}
    for seq_name in og_summary:
        bd_rate = bd_rate_summaries(og_summary[seq_name], run_summaries[seq_name])
        results[seq_name] = bd_rate
    print_md_table(results)

    # gen_rd_plots([sum for sums in og_summary.values() for sum in sums])
    some_images = [f"kodim{num:02}" for num in range(1, 9)]
    gen_rd_plots(
        [sum for seq_name in some_images for sum in og_summary[seq_name]],
        [sum for seq_name in some_images for sum in run_summaries[seq_name]],
    )
    plt.show()

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from coolchic.eval.bd_rate import (
    SummaryEncodingMetrics,
    full_run_summary,
    parse_result_summary,
)
from coolchic.eval.plotting import gen_rd_iterations

all_runs = Path("results/exps/copied/only_latents/")
all_runs = Path("results/exps/copied/only_latents/")
og_summary_dir = Path("results/image/kodak/results.tsv")
hm_summary_dir = Path("results/image/kodak/hm.tsv")
jpeg_summary_dir = Path("results/image/kodak/jpeg.tsv")

all_anchors = {
    "cool-chic": og_summary_dir,
    "hm": hm_summary_dir,
    "jpeg": jpeg_summary_dir,
}

real_rate = False

og_summary = parse_result_summary(og_summary_dir)


def get_summaries_from_sweep(
    all_runs: Path, chosen_img: str, real_rate: bool = False
) -> list[SummaryEncodingMetrics]:
    all_data = []
    for run_path in tqdm(list(all_runs.iterdir())):
        # Let's check that the results come from a sweep.
        sub_runs = list(run_path.iterdir())
        if not any(
            [
                sub_run.is_dir() and "kodim_config" in str(sub_run)
                for sub_run in sub_runs
            ]
        ):
            continue
        run_summaries = full_run_summary(run_path, real_rate=real_rate)
        n_itr = list(run_summaries.values())[0][0].n_itr
        n_train_loops = list(run_summaries.values())[0][0].n_train_loops
        # All runs in this dir have the same n_itr and n_train_loops, but let's make sure.
        assert all(
            config.n_itr == n_itr and config.n_train_loops == n_train_loops
            for img_configs in run_summaries.values()
            for config in img_configs
        )
        all_data.extend(run_summaries[chosen_img])
    return all_data


if __name__ == "__main__":
    # gen_rd_plots(
    #     og_summary[chosen_img],
    #     get_summaries_from_sweep(all_runs, chosen_img, real_rate=real_rate),
    # )
    def get_df_1_img(chosen_img):
        og_df = pd.DataFrame([s.model_dump() for s in og_summary[chosen_img]])
        all_summaries = get_summaries_from_sweep(
            all_runs, chosen_img, real_rate=real_rate
        )
        sum_df = pd.DataFrame([s.model_dump() for s in all_summaries])
        sum_df = sum_df[sum_df.seq_name == chosen_img]
        assert isinstance(sum_df, pd.DataFrame)
        return og_df, sum_df

    for img in [f"kodim{i:02d}" for i in range(1, 10)]:
        og_df, sum_df = get_df_1_img(img)
        fig, ax = gen_rd_iterations(og_df, sum_df)
    plt.show()

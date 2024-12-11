import argparse
import shutil
from gen_kodim_config import save_cfg
import itertools
from pathlib import Path

params_1 = {
    "n_itr": [1000, 2500, 5000, 7500, 10000, 12000],
    "n_train_loops": [1, 2, 3, 5],
}
params_2 = {
    "n_itr": [15_000, 20_000, 30_000, 40_000, 50_000, 75_000, 100_000],
    "n_train_loops": [1],
}
params_to_search = params_2


def estimate_time(params: dict[str, list[int]], n_gpus: int = 8) -> float:
    all_combs = itertools.product(*params.values())

    def time_for_one_run(n_it: int, n_loops: int):
        time_one_run = (n_it / 10000) * 10 * n_loops
        # We need to run every param combination for all lambdas and images.
        time_all = time_one_run * 120
        return time_all

    return sum(time_for_one_run(*run) for run in all_combs) / n_gpus


def replace_params(template: str, chosen_params: dict[str, str]) -> str:
    """Replace the placeholders in the template by the chosen parameters.
    The parameters must appear in the template between curly brackets.
    """
    for param in chosen_params:
        placeholder = (
            f"{{{param}}}"  # doubling the sq brackets yields a sq bracket in f-strs.
        )
        if placeholder not in template:
            raise ValueError(
                f"Expected a placeholder for {param=} in the template but wasn't found."
            )
        template = template.replace(placeholder, chosen_params[param])
    return template


def get_dir_name(chosen_params: dict[str, str]) -> str:
    name = ""
    first = True
    for param in chosen_params:
        if not first:
            name += "_"
        first = False
        name += f"{param}-{chosen_params[param]}"
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--remove-prev", action="store_true")
    args = parser.parse_args()

    template_path: Path = args.template
    assert template_path.exists()
    exps_dir = template_path.parent
    template_text = template_path.read_text()

    # Remove all subdirectories if specifically required.
    if args.remove_prev:
        for dir in exps_dir.iterdir():
            if dir.is_dir():
                shutil.rmtree(dir)

    param_keys = list(params_to_search.keys())
    for comb in itertools.product(*[params_to_search[param] for param in param_keys]):
        chosen_comb = {param: str(comb[i]) for i, param in enumerate(param_keys)}
        text = replace_params(template_text, chosen_comb)

        dir_name = get_dir_name(chosen_comb)
        new_dir = exps_dir / dir_name
        new_dir.mkdir(exist_ok=True)
        save_cfg(text, new_dir, "config_template", template=True)

    # Estimate runtime of the whole suite.
    time_lower_bound = estimate_time(params_to_search, n_gpus=8)
    print(
        f"Estimated lower bound for the runtime (Wall): {time_lower_bound} minutes, or {time_lower_bound/60:.2f} hours."
    )

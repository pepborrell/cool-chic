from pathlib import Path

import yaml


def get_best_model(
    best_model_dir=Path("results/exps/n_it-grid/n_itr-100000_n_train_loops-1/"),
) -> dict[tuple[str, float], Path]:
    """Computes a map with the best model that was trained for a given image and lambda value."""
    all_jobs = [dir for dir in best_model_dir.iterdir() if dir.is_dir()]
    all_runs = [run for dir in all_jobs for run in dir.iterdir() if run.is_dir()]
    all_configs = [run / "param.txt" for run in all_runs]

    params_to_path = {}

    for config_path in all_configs:
        cfg = yaml.unsafe_load(config_path.read_text())

        img_name = cfg["input"].stem
        lmbda = cfg["lmbda"]
        params_to_path[(img_name, lmbda)] = config_path.parent

    return params_to_path


if __name__ == "__main__":
    print(get_best_model())

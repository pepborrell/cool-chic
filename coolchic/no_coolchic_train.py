import argparse
import os
from pathlib import Path

import torch

import wandb
from coolchic.enc.utils.misc import get_best_device
from coolchic.hypernet.hypernet import NOWholeNet
from coolchic.hypernet.inference import eval_on_all_kodak
from coolchic.hypernet.training import train
from coolchic.metalearning.data import OpenImagesDataset
from coolchic.utils.paths import COOLCHIC_REPO_ROOT
from coolchic.utils.structs import ConstantIterable
from coolchic.utils.types import (
    HypernetRunConfig,
    load_config,
)


def get_workdir_hypernet(config: HypernetRunConfig, config_path: Path) -> Path:
    workdir = (
        config.workdir
        if config.workdir is not None
        # If no workdir is specified, results will be saved in results/{path_to_config_relative_to_cfg}/
        else COOLCHIC_REPO_ROOT
        / "results"
        / config_path.relative_to("cfg").with_suffix("")
    )
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def main():
    # Configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Specifies the path to the config file that will be used."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    run_cfg = load_config(config_path, HypernetRunConfig)
    workdir = get_workdir_hypernet(run_cfg, config_path)

    # Automatic device detection
    device = get_best_device()
    print(f'{"Device":<20}: {device}')

    # Load data
    train_data = OpenImagesDataset(
        run_cfg.n_samples, patch_size=run_cfg.hypernet_cfg.patch_size, train=True
    )
    test_data = OpenImagesDataset(run_cfg.n_samples, patch_size=None, train=False)
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size=run_cfg.batch_size, shuffle=False
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False
    )

    # Lambda definition logic.
    if isinstance(run_cfg.lmbda, float):
        lmbdas = ConstantIterable(run_cfg.lmbda)
    elif run_cfg.lmbda == "random":
        raise NotImplementedError("Random lambda not implemented.")
        # In coolchic experiments, we ran these lambda values: [0.0001, 0.0004, 0.001, 0.004, 0.02].
        lmbda_min, lmbda_max = 0.0001, 0.02
        lmbdas = (
            torch.rand(run_cfg.n_samples) * (lmbda_max - lmbda_min) + lmbda_min
        ).tolist()
    else:
        raise ValueError(f"Invalid lambda value: {run_cfg.lmbda}")

    ##### LOGGING #####
    # Setting up all logging using wandb.
    if run_cfg.disable_wandb:
        # To disable wandb completely.
        os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_MODE"] = "online"
    # Start wandb logging.
    wandb_run = wandb.init(project="coolchic-runs", config=run_cfg.model_dump())

    ##### INSTANTIATE MODEL #####
    model = NOWholeNet(run_cfg.hypernet_cfg)

    # Train
    net = train(
        train_data=train_data_loader,
        test_data=test_data_loader,
        wholenet=model,
        n_epochs=run_cfg.n_epochs,
        lmbdas=lmbdas,
        unfreeze_backbone_samples=run_cfg.unfreeze_backbone,
        workdir=workdir,
        device=device,
    )

    # Eval on kodak at end of training.
    kodak_df = eval_on_all_kodak(net, lmbda=run_cfg.lmbda)
    kodak_df["lmbda"] = run_cfg.lmbda
    kodak_df["anchor"] = "hypernet"
    kodak_df.to_csv(workdir / "kodak_results.csv")
    wandb_run.log({"kodak_results": wandb.Table(dataframe=kodak_df)})


if __name__ == "__main__":
    main()

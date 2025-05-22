import argparse
import os
from pathlib import Path

import torch
import wandb

from coolchic.enc.utils.misc import get_best_device
from coolchic.hypernet.hypernet import NOWholeNet
from coolchic.hypernet.training import get_workdir_hypernet, train
from coolchic.metalearning.data import OpenImagesDataset
from coolchic.utils.paths import get_latest_checkpoint
from coolchic.utils.types import (
    HypernetRunConfig,
    load_config,
)


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
    print(f"{device=}")

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

    ##### LOGGING #####
    # Setting up all logging using wandb.
    if run_cfg.disable_wandb:
        # To disable wandb completely.
        os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_MODE"] = "online"
    # Start wandb logging.
    wandb.init(project="coolchic-runs", config=run_cfg.model_dump())

    ##### INSTANTIATE MODEL #####
    model = NOWholeNet(run_cfg.hypernet_cfg)

    if run_cfg.checkpoint is not None:
        # If a checkpoint is given, we use it as init, then we keep training it.
        # This option takes precedence over the NO coolchic init.
        if run_cfg.checkpoint.stem == "__latest":
            run_cfg.checkpoint = get_latest_checkpoint(run_cfg.checkpoint.parent)
        model.load_state_dict(torch.load(run_cfg.checkpoint, weights_only=True))
        # We need to know which iteration the checkpoint was at.
        chckpoint_it_number = int(run_cfg.checkpoint.stem.split("_")[-1])
    else:
        chckpoint_it_number = None

    # Train
    _ = train(
        train_data=train_data_loader,
        test_data=test_data_loader,
        wholenet=model,
        recipe=run_cfg.recipe,
        lmbda=run_cfg.lmbda,
        unfreeze_backbone_samples=run_cfg.unfreeze_backbone,
        workdir=workdir,
        device=device,
        checkpoint_samples=chckpoint_it_number,
    )


if __name__ == "__main__":
    main()

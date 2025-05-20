import argparse
import os
from pathlib import Path

import torch

import wandb
from coolchic.enc.utils.misc import get_best_device
from coolchic.hypernet.hypernet import (
    DeltaWholeNet,
    NOWholeNet,
    SmallDeltaWholeNet,
)
from coolchic.hypernet.inference import eval_on_all_kodak, load_hypernet
from coolchic.hypernet.training import get_workdir_hypernet, train
from coolchic.metalearning.data import OpenImagesDataset
from coolchic.utils.paths import get_latest_checkpoint
from coolchic.utils.types import HypernetRunConfig, load_config


def main():
    # Configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Specifies the path to the config file that will be used."
    )
    parser.add_argument("--small", action="store_true", help="Use small model.")
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
        train_data, batch_size=run_cfg.batch_size, shuffle=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False
    )

    #### LOADING HYPERNET ####
    if args.small:
        wholenet = SmallDeltaWholeNet(run_cfg.hypernet_cfg)
    else:
        wholenet = DeltaWholeNet(run_cfg.hypernet_cfg)
    wholenet = wholenet.to(device)

    chckpoint_it_number = None
    if run_cfg.checkpoint is not None:
        # If a checkpoint is given, we use it as init, then we keep training it.
        # This option takes precedence over the NO coolchic init.
        if run_cfg.checkpoint.stem == "__latest":
            run_cfg.checkpoint = get_latest_checkpoint(run_cfg.checkpoint.parent)
        wholenet.load_state_dict(torch.load(run_cfg.checkpoint, weights_only=True))
        # We need to know which iteration the checkpoint was at.
        chckpoint_it_number = int(run_cfg.checkpoint.stem.split("_")[-1])
        assert wholenet.use_delta, "Model must be using deltas."
    elif run_cfg.model_weights is not None:
        # If N-O coolchic model is given, we use it as init.
        # If model has name latest, take the latest in the folder.
        if run_cfg.model_weights.stem == "__latest":
            run_cfg.model_weights = get_latest_checkpoint(run_cfg.model_weights.parent)

        assert (
            run_cfg.model_weights.exists()
        ), "Specified model weights path doesn't exist."
        assert (
            run_cfg.model_weights.suffix == ".pt"
        ), "Model weights must be a .pt file."
        print(f"Loading model weights from {run_cfg.model_weights}")
        no_model = load_hypernet(
            weights_path=run_cfg.model_weights, config=run_cfg, wholenet_cls=NOWholeNet
        ).to(device)
        wholenet.load_from_no_coolchic(no_model)
    else:
        # We don't want to train this from scratch anymore.
        raise ValueError("Model weights must be provided.")

    ##### LOGGING #####
    # Setting up all logging using wandb.
    if run_cfg.disable_wandb:
        # To disable wandb completely.
        os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_MODE"] = "online"
    # Start wandb logging.
    wandb_run = wandb.init(project="coolchic-runs", config=run_cfg.model_dump())

    # Train
    net = train(
        train_data_loader,
        test_data_loader,
        wholenet=wholenet,
        recipe=run_cfg.recipe,
        lmbda=run_cfg.lmbda,
        unfreeze_backbone_samples=run_cfg.unfreeze_backbone,
        workdir=workdir,
        device=device,
        checkpoint_samples=chckpoint_it_number,
    )

    # Eval on kodak at end of training.
    kodak_df = eval_on_all_kodak(net, lmbda=run_cfg.lmbda, mlp_rate=False)
    kodak_df["lmbda"] = run_cfg.lmbda
    kodak_df["anchor"] = "hypernet"
    kodak_df.to_csv(workdir / "kodak_results.csv")
    wandb_run.log({"kodak_results": wandb.Table(dataframe=kodak_df)})


if __name__ == "__main__":
    main()

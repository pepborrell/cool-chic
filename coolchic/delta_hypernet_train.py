import argparse
import os
from pathlib import Path

import torch

import wandb
from coolchic.enc.component.coolchic import CoolChicEncoderOutput
from coolchic.enc.training.loss import LossFunctionOutput, loss_function
from coolchic.enc.utils.misc import POSSIBLE_DEVICE, get_best_device
from coolchic.hypernet.hypernet import DeltaWholeNet, NOWholeNet
from coolchic.hypernet.inference import eval_on_all_kodak, load_hypernet
from coolchic.hypernet.training import train
from coolchic.metalearning.data import OpenImagesDataset
from coolchic.utils.paths import COOLCHIC_REPO_ROOT
from coolchic.utils.structs import ConstantIterable
from coolchic.utils.types import HypernetRunConfig, load_config


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


def get_mlp_rate(net: DeltaWholeNet) -> float:
    rate_mlp = 0.0
    rate_per_module = net.mean_decoder.get_network_rate()
    for _, module_rate in rate_per_module.items():  # pyright: ignore
        for _, param_rate in module_rate.items():  # weight, bias
            rate_mlp += param_rate
    return rate_mlp


def evaluate_wholenet(
    net: DeltaWholeNet,
    test_data: torch.utils.data.DataLoader,
    lmbda: float,
    device: POSSIBLE_DEVICE,
) -> dict[str, float]:
    net.eval()
    with torch.no_grad():
        all_losses: list[LossFunctionOutput] = []
        for test_img in test_data:
            test_img = test_img.to(device)
            raw_out, rate, add_data = net.forward(
                test_img,
                quantizer_noise_type="none",
                quantizer_type="hardround",
            )
            test_out = CoolChicEncoderOutput(
                raw_out=raw_out, rate=rate, additional_data=add_data
            )
            rate_mlp = get_mlp_rate(net)
            test_loss = loss_function(
                test_out.raw_out,
                test_out.rate,
                test_img,
                lmbda=lmbda,
                rate_mlp_bit=rate_mlp,
                compute_logs=True,
            )
            all_losses.append(test_loss)

        loss_tensor = torch.stack([loss.loss for loss in all_losses])  # pyright: ignore
        avg_loss = torch.mean(loss_tensor).item()
        std_loss = torch.std(loss_tensor).item()
        avg_mse = torch.mean(torch.tensor([loss.mse for loss in all_losses])).item()
        avg_psnr_db = torch.mean(
            torch.tensor([loss.psnr_db for loss in all_losses])
        ).item()
        avg_total_rate_bpp = torch.mean(
            torch.tensor([loss.total_rate_bpp for loss in all_losses])
        ).item()
    # Switch back to training mode.
    net.train()
    return {
        "test_loss": avg_loss,
        "test_mse": avg_mse,
        "test_psnr_db": avg_psnr_db,
        "test_total_rate_bpp": avg_total_rate_bpp,
        "std_test_loss": std_loss,
    }


def print_gpu_info():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


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

    #### LOADING HYPERNET ####
    wholenet = DeltaWholeNet(run_cfg.hypernet_cfg)
    if run_cfg.model_weights is not None:
        # If N-O coolchic model is given, we use it as init.

        # If model has name latest, take the latest in the folder.
        if run_cfg.model_weights.stem == "__latest":
            # Weights formatted like epoch_7_batch_3550000.pt, we take the highest sample number.
            checkpoints = [
                file
                for file in run_cfg.model_weights.parent.iterdir()
                if file.suffix == ".pt"
            ]
            run_cfg.model_weights = max(
                checkpoints, key=lambda p: int(p.stem.split("_")[-1])
            )

        assert (
            run_cfg.model_weights.exists()
        ), "Specified model weights path doesn't exist."
        print(f"Loading model weights from {run_cfg.model_weights}")
        no_model = load_hypernet(
            weights_path=run_cfg.model_weights, config=run_cfg, wholenet_cls=NOWholeNet
        )
        wholenet.load_from_no_coolchic(no_model)
    else:
        # We don't want to train this from scratch anymore.
        raise ValueError("Model weights must be provided.")

    # The part that comes from NO coolchic is already trained.
    # Let's freeze that.
    for param in wholenet.mean_decoder.parameters():
        param.requires_grad = False

    # Lambda definition logic.
    if isinstance(run_cfg.lmbda, float):
        lmbdas = ConstantIterable(run_cfg.lmbda)
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

    # Train
    net = train(
        train_data_loader,
        test_data_loader,
        wholenet=wholenet,
        n_epochs=run_cfg.n_epochs,
        lmbdas=lmbdas,
        unfreeze_backbone_samples=run_cfg.unfreeze_backbone,
        start_lr=run_cfg.start_lr,
        workdir=workdir,
        device=device,
        softround_temperature=run_cfg.softround_temperature,
        noise_parameter=run_cfg.noise_parameter,
    )

    # Eval on kodak at end of training.
    kodak_df = eval_on_all_kodak(net, lmbda=run_cfg.lmbda)
    kodak_df["lmbda"] = run_cfg.lmbda
    kodak_df["anchor"] = "hypernet"
    kodak_df.to_csv(workdir / "kodak_results.csv")
    wandb_run.log({"kodak_results": wandb.Table(dataframe=kodak_df)})


if __name__ == "__main__":
    main()

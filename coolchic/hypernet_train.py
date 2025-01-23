import argparse
import os
from pathlib import Path

import torch
import yaml

import wandb
from coolchic.enc.component.coolchic import CoolChicEncoderOutput
from coolchic.enc.training.loss import LossFunctionOutput, loss_function
from coolchic.enc.utils.misc import POSSIBLE_DEVICE, get_best_device
from coolchic.hypernet.hypernet import CoolchicWholeNet
from coolchic.metalearning.data import OpenImagesDataset
from coolchic.utils.paths import COOLCHIC_REPO_ROOT
from coolchic.utils.types import HyperNetConfig, HypernetRunConfig


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


def get_mlp_rate(net: CoolchicWholeNet) -> float:
    rate_mlp = 0.0
    rate_per_module = net.cc_encoder.get_network_rate()
    for _, module_rate in rate_per_module.items():  # pyright: ignore
        for _, param_rate in module_rate.items():  # weight, bias
            rate_mlp += param_rate
    return rate_mlp


def evaluate_wholenet(
    net: CoolchicWholeNet,
    test_data: torch.utils.data.DataLoader,
    lmbda: float,
    device: POSSIBLE_DEVICE,
) -> dict[str, float]:
    all_losses: list[LossFunctionOutput] = []
    for test_img in test_data:
        test_img = test_img.to(device)
        raw_out, rate, add_data = net.forward(test_img)
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
    avg_total_rate_bpp = torch.mean(
        torch.tensor([loss.total_rate_bpp for loss in all_losses])
    ).item()
    return {
        "test_loss": avg_loss,
        "test_mse": avg_mse,
        "test_total_rate_bpp": avg_total_rate_bpp,
        "std_test_loss": std_loss,
    }


def print_gpu_info():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def train(
    train_data: torch.utils.data.DataLoader,
    test_data: torch.utils.data.DataLoader,
    config: HyperNetConfig,
    n_epochs: int,
    lmbda: float,
    workdir: Path,
    device: POSSIBLE_DEVICE,
    start_lr: float = 1e-3,
):
    wholenet = CoolchicWholeNet(config)
    if torch.cuda.is_available():
        print_gpu_info()
        wholenet = wholenet.cuda()
    else:
        wholenet.to(device)
    optimizer = torch.optim.Adam(wholenet.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_epochs, eta_min=1e-5
    )

    train_losses = []

    wholenet.freeze_resnet()
    for epoch in range(n_epochs):
        if epoch > 5:
            wholenet.unfreeze_resnet()
        batch_n = 0
        for img_batch in train_data:
            img_batch = img_batch.to(device)
            raw_out, rate, add_data = wholenet.forward(img_batch)
            out_forward = CoolChicEncoderOutput(
                raw_out=raw_out, rate=rate, additional_data=add_data
            )
            loss_function_output = loss_function(
                out_forward.raw_out,
                out_forward.rate,
                img_batch,
                lmbda=lmbda,
                rate_mlp_bit=0.0,
                compute_logs=True,
            )
            # Logging training numbers.
            assert isinstance(
                loss_function_output.loss, torch.Tensor
            ), "Loss is not a tensor"
            train_losses.append(
                {
                    "epoch": epoch,
                    "batch": batch_n,
                    "train_loss": loss_function_output.loss.item(),
                    "train_mse": loss_function_output.mse,
                    "train_total_rate_bpp": loss_function_output.total_rate_bpp,
                    "train_psnr_db": loss_function_output.psnr_db,
                }
            )
            total_loss = loss_function_output.loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_n += 1

            if batch_n % 20 == 0:
                # Average train losses.
                train_losses_avg = {
                    "train_loss": torch.mean(
                        torch.tensor([loss["train_loss"] for loss in train_losses])
                    ),
                    "train_mse": torch.mean(
                        torch.tensor([loss["train_mse"] for loss in train_losses])
                    ),
                    "train_total_rate_bpp": torch.mean(
                        torch.tensor(
                            [loss["train_total_rate_bpp"] for loss in train_losses]
                        )
                    ),
                }
                train_losses = []
                # Evaluate on test data
                eval_results = evaluate_wholenet(
                    wholenet, test_data, lmbda=lmbda, device=device
                )
                print(eval_results)
                wandb.log(
                    {
                        "epoch": epoch,
                        "batch": batch_n,
                        **train_losses_avg,
                        **eval_results,
                    }
                )

                # Save model
                save_path = workdir / f"epoch_{epoch}_batch_{batch_n}.pt"
                torch.save(wholenet.state_dict(), save_path)

        scheduler.step()

    return wholenet


def main():
    # Configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Specifies the path to the config file that will be used."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as stream:
        run_cfg = HypernetRunConfig(**yaml.safe_load(stream))
    workdir = get_workdir_hypernet(run_cfg, config_path)

    # Automatic device detection
    device = get_best_device()
    print(f'{"Device":<20}: {device}')

    # Load data
    train_data = OpenImagesDataset(
        run_cfg.n_samples,
        patch_size=run_cfg.hypernet_cfg.patch_size,
        train=True,
        check_downloaded=False,
    )
    test_data = OpenImagesDataset(
        run_cfg.n_samples, patch_size=None, train=False, check_downloaded=False
    )
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

    # Train
    _ = train(
        train_data_loader,
        test_data_loader,
        config=run_cfg.hypernet_cfg,
        n_epochs=run_cfg.n_epochs,
        lmbda=run_cfg.lmbda,
        workdir=workdir,
        device=device,
    )


if __name__ == "__main__":
    main()

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

import wandb
from coolchic.enc.training.loss import loss_function
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
        / config.unique_id  # unique id to distinguish different runs launched by the same config.
    )
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def get_mlp_rate(net: CoolchicWholeNet) -> float:
    rate_mlp = 0.0
    rate_per_module = net.cc_encoder.get_network_rate()
    for _, module_rate in asdict(rate_per_module).items():
        for _, param_rate in module_rate.items():  # weight, bias
            rate_mlp += param_rate
    return rate_mlp


def evaluate_wholenet(
    net: CoolchicWholeNet, test_data, lmbda: float
) -> dict[str, float]:
    all_losses = []
    for test_img in test_data:
        test_out = net(test_img)
        rate_mlp = get_mlp_rate(net)
        test_loss = loss_function(
            test_out.decoded_image,
            test_out.rate,
            test_img,
            lmbda=lmbda,
            rate_mlp_bit=rate_mlp,
            compute_logs=False,
        )
        all_losses.append(test_loss)

    avg_loss = torch.mean(torch.stack([loss.loss for loss in all_losses])).item()
    avg_mse = torch.mean(torch.stack([loss.mse for loss in all_losses])).item()
    avg_total_rate_bpp = torch.mean(
        torch.stack([loss.total_rate_bpp for loss in all_losses])
    ).item()
    std_loss = torch.std(torch.stack([loss.loss for loss in all_losses])).item()
    return {
        "test_loss": avg_loss,
        "test_mse": avg_mse,
        "test_total_rate_bpp": avg_total_rate_bpp,
        "std_test_loss": std_loss,
    }


def train(
    train_data,
    test_data,
    config: HyperNetConfig,
    n_epochs: int,
    lmbda: float,
    workdir: Path,
    device: POSSIBLE_DEVICE,
):
    wholenet = CoolchicWholeNet(config)
    wholenet.to(device)
    optimizer = torch.optim.Adam(wholenet.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        for i, img in enumerate(train_data):
            out_forward = wholenet(img)
            loss_function_output = loss_function(
                out_forward.decoded_image,
                out_forward.rate,
                img,
                lmbda=lmbda,
                rate_mlp_bit=0.0,
                compute_logs=False,
            )
            optimizer.zero_grad()
            assert isinstance(
                loss_function_output.loss, torch.Tensor
            ), "Loss is not a tensor"
            loss_function_output.loss.backward()
            optimizer.step()

            if i % 20 == 0:
                # Evaluate on test data
                eval_results = evaluate_wholenet(wholenet, test_data, lmbda=lmbda)
                print(f"Epoch {epoch}, iteration {i}:")
                print(eval_results)
                wandb.log({"epoch": epoch, "iteration": i, **eval_results})

                # Save model
                save_path = workdir / f"epoch_{epoch}_it_{i}.pt"
                torch.save(wholenet.state_dict(), save_path)

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
    all_data = OpenImagesDataset(run_cfg.n_samples, device=device)
    n_train = int(len(all_data) * 0.8)
    train_data = all_data[:n_train]
    test_data = all_data[n_train:]

    # Train
    _ = train(
        train_data,
        test_data,
        config=run_cfg.hypernet_cfg,
        n_epochs=run_cfg.n_epochs,
        lmbda=run_cfg.lmbda,
        workdir=workdir,
        device=device,
    )


if __name__ == "__main__":
    main()

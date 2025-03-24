import argparse
import os
from pathlib import Path
from typing import Iterable

import torch

import wandb
from coolchic.enc.component.coolchic import CoolChicEncoderOutput
from coolchic.enc.training.loss import LossFunctionOutput, loss_function
from coolchic.enc.training.quantizemodel import quantize_model
from coolchic.enc.utils.misc import POSSIBLE_DEVICE, get_best_device
from coolchic.hypernet.hypernet import NOWholeNet
from coolchic.hypernet.inference import eval_on_all_kodak
from coolchic.metalearning.data import OpenImagesDataset
from coolchic.utils.nn import _linear_schedule, get_mlp_rate
from coolchic.utils.paths import COOLCHIC_REPO_ROOT
from coolchic.utils.structs import ConstantIterable
from coolchic.utils.types import HyperNetConfig, HypernetRunConfig, load_config


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


def evaluate_wholenet(
    net: NOWholeNet,
    test_data: torch.utils.data.DataLoader,
    lmbda: float,
    device: POSSIBLE_DEVICE,
) -> dict[str, float]:
    net.eval()
    with torch.no_grad():
        all_losses: list[LossFunctionOutput] = []
        rate_mlp: float | None = None
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

            if not rate_mlp:
                # getting mlp rate involves "mocking" a model quantization.
                cc_enc = net.image_to_coolchic(test_img, stop_grads=True).cpu()
                cc_enc._store_full_precision_param()
                cc_enc = quantize_model(encoder=cc_enc, input_img=test_img, lmbda=lmbda)
                rate_mlp = get_mlp_rate(cc_enc)

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


def train(
    train_data: torch.utils.data.DataLoader,
    test_data: torch.utils.data.DataLoader,
    config: HyperNetConfig,
    n_epochs: int,
    lmbdas: Iterable[float],
    workdir: Path,
    device: POSSIBLE_DEVICE,
    unfreeze_backbone_samples: int,
    start_lr: float = 1e-3,
    softround_temperature: tuple[float, float] = (0.3, 0.3),
    noise_parameter: tuple[float, float] = (0.25, 0.25),
):
    wholenet = NOWholeNet(config)
    if torch.cuda.is_available():
        print_gpu_info()
        wholenet = wholenet.cuda()
    else:
        wholenet = wholenet.to(device)
    optimizer = torch.optim.Adam(wholenet.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_epochs, eta_min=1e-6
    )
    batch_size = train_data.batch_size
    assert batch_size is not None, "Batch size must be set."
    total_iterations = len(train_data) * n_epochs * batch_size

    train_losses = []
    samples_seen = 0

    wholenet.freeze_resnet()
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        batch_n = 0
        for img_batch, lmbda in zip(train_data, lmbdas):
            img_batch = img_batch.to(device)
            cur_softround_t = _linear_schedule(
                *softround_temperature, samples_seen, total_iterations
            )
            cur_noise_param = _linear_schedule(
                *noise_parameter, samples_seen, total_iterations
            )
            raw_out, rate, add_data = wholenet.forward(
                img_batch,
                softround_temperature=cur_softround_t,
                noise_parameter=cur_noise_param,
            )
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
                    "samples_seen": samples_seen,
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
            # Clip gradients to avoid exploding gradients.
            torch.nn.utils.clip_grad_norm_(wholenet.parameters(), 1.0)
            optimizer.step()

            batch_n += 1
            samples_seen += batch_size

            if (samples_seen % 500) < batch_size:
                # Average train losses.
                train_losses_avg = {
                    "train_loss": torch.mean(
                        torch.tensor([loss["train_loss"] for loss in train_losses])
                    ),
                    "train_mse": torch.mean(
                        torch.tensor([loss["train_mse"] for loss in train_losses])
                    ),
                    "train_psnr_db": torch.mean(
                        torch.tensor([loss["train_psnr_db"] for loss in train_losses])
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

                # Save model, but only every 10k batches.
                if (samples_seen % 10000) < batch_size:
                    save_path = workdir / f"epoch_{epoch}_batch_{samples_seen}.pt"
                    torch.save(wholenet.state_dict(), save_path)

                # Unfreeze backbone if needed
                if samples_seen > unfreeze_backbone_samples:
                    wholenet.unfreeze_resnet()
                    print("Unfreezing backbone")

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

    # Train
    net = train(
        train_data_loader,
        test_data_loader,
        config=run_cfg.hypernet_cfg,
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
    kodak_df = eval_on_all_kodak(net)
    kodak_df["lmbda"] = run_cfg.lmbda
    kodak_df["anchor"] = "hypernet"
    kodak_df.to_csv(workdir / "kodak_results.csv")
    wandb_run.log({"kodak_results": wandb.Table(dataframe=kodak_df)})


if __name__ == "__main__":
    main()

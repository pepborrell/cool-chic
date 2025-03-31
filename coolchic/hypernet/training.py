from pathlib import Path
from typing import Iterable

import torch

import wandb
from coolchic.enc.component.coolchic import CoolChicEncoderOutput
from coolchic.enc.component.core.quantizer import POSSIBLE_QUANTIZATION_NOISE_TYPE
from coolchic.enc.training.loss import LossFunctionOutput, loss_function
from coolchic.enc.training.quantizemodel import quantize_model
from coolchic.enc.utils.misc import POSSIBLE_DEVICE
from coolchic.hypernet.hypernet import WholeNet
from coolchic.utils.nn import _linear_schedule, get_mlp_rate


def evaluate_wholenet(
    net: WholeNet,
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

            # getting mlp rate involves "mocking" a model quantization.
            cc_enc = net.image_to_coolchic(test_img, stop_grads=True)
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


def train(
    train_data: torch.utils.data.DataLoader,
    test_data: torch.utils.data.DataLoader,
    wholenet: WholeNet,
    n_epochs: int,
    lmbdas: Iterable[float],
    workdir: Path,
    device: POSSIBLE_DEVICE,
    unfreeze_backbone_samples: int,
    start_lr: float = 1e-3,
    softround_temperature: tuple[float, float] = (0.3, 0.3),
    noise_parameter: tuple[float, float] = (0.25, 0.25),
    quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "gaussian",
):
    wholenet = wholenet.to(device)
    optimizer = torch.optim.Adam(wholenet.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_data), eta_min=1e-6
    )

    batch_size = train_data.batch_size
    assert batch_size is not None, "Batch size is None"
    total_iterations = len(train_data) * batch_size * n_epochs

    train_losses = []
    samples_seen = 0
    best_model = wholenet.state_dict()
    best_test_loss = float("inf")

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
                quantizer_noise_type=quantizer_noise_type,
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
            scheduler.step()

            # In NO coolchic, this means logging roughly every 5 mins.
            if (samples_seen % 5000) < batch_size:
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
                        "samples_seen": samples_seen,
                        "n_iterations": samples_seen // batch_size,
                        **train_losses_avg,
                        **eval_results,
                    }
                )

                # Patience: save model every 50k samples if it's better than the best model so far.
                # In NO coolchic this happens every hour.
                if (samples_seen % 50000) < batch_size:
                    # Only save if it's better than the best model so far.
                    if eval_results["test_loss"] < best_test_loss:
                        best_model = wholenet.state_dict()
                        best_test_loss = eval_results["test_loss"]
                        save_path = workdir / f"epoch_{epoch}_batch_{samples_seen}.pt"
                        torch.save(wholenet.state_dict(), save_path)
                    else:
                        # Reset to last best model.
                        wholenet.load_state_dict(best_model)

                # Unfreeze backbone if needed
                if samples_seen > unfreeze_backbone_samples:
                    wholenet.unfreeze_resnet()
                    print("Unfreezing backbone")

        scheduler.step()

    return wholenet

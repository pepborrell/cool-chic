from pathlib import Path

import torch
from pydantic import BaseModel

import wandb
from coolchic.enc.component.coolchic import CoolChicEncoderOutput
from coolchic.enc.training.loss import LossFunctionOutput, loss_function
from coolchic.enc.training.quantizemodel import quantize_model
from coolchic.enc.utils.misc import POSSIBLE_DEVICE
from coolchic.hypernet.hypernet import WholeNet
from coolchic.utils.nn import _linear_schedule, get_mlp_rate
from coolchic.utils.paths import COOLCHIC_REPO_ROOT
from coolchic.utils.types import HypernetRunConfig, PresetConfig


def cycle(iterable):
    """Cycle through an iterable indefinitely.
    Implemented here because itertools.cycle saves all elements,
    introducing memory leaks.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class RunningTrainLoss(BaseModel):
    run_loss: float = 0
    run_mse: float = 0
    run_total_rate_bpp: float = 0
    run_psnr_db: float = 0
    n_samples: int = 0

    def add(self, loss_output: LossFunctionOutput) -> None:
        # To make pyright happy.
        assert isinstance(loss_output.loss, torch.Tensor), "Loss is not a tensor"
        assert loss_output.mse is not None, "MSE is None"
        assert loss_output.total_rate_bpp is not None, "Total rate bpp is None"
        assert loss_output.psnr_db is not None, "PSNR db is None"

        self.run_loss += loss_output.loss.detach().cpu().item()
        self.run_mse += loss_output.mse
        self.run_total_rate_bpp += loss_output.total_rate_bpp
        self.run_psnr_db += loss_output.psnr_db
        self.n_samples += 1

    def average(self) -> dict[str, float]:
        return {
            "train_loss": self.run_loss / self.n_samples,
            "train_mse": self.run_mse / self.n_samples,
            "train_total_rate_bpp": self.run_total_rate_bpp / self.n_samples,
            "train_psnr_db": self.run_psnr_db / self.n_samples,
        }


def evaluate_wholenet(
    net: WholeNet,
    test_data: torch.utils.data.DataLoader,
    lmbda: float,
    device: POSSIBLE_DEVICE,
) -> dict[str, float]:
    net.eval()
    with torch.no_grad():
        all_losses: list[float] = []
        all_mse_losses: list[float] = []
        all_psnr_db: list[float] = []
        all_total_rate_bpp: list[float] = []

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

            # The MLP rate is very roughly the same for all images, so we calculate it
            # once for the whole batch.
            if not rate_mlp:
                # getting mlp rate involves "mocking" a model quantization.
                cc_enc = net.image_to_coolchic(test_img, stop_grads=True)
                cc_enc._store_full_precision_param()
                cc_enc = quantize_model(encoder=cc_enc, input_img=test_img, lmbda=lmbda)
                rate_mlp = get_mlp_rate(cc_enc)
                del cc_enc  # I suspect cc_enc has memory leaks, so we delete it.
                torch.cuda.empty_cache()

            test_loss = loss_function(
                test_out.raw_out,
                test_out.rate,
                test_img,
                lmbda=lmbda,
                rate_mlp_bit=rate_mlp,
                compute_logs=True,
            )
            assert isinstance(test_loss.loss, torch.Tensor), "Loss is not a tensor"
            all_losses.append(test_loss.loss.detach().cpu().item())
            assert (
                test_loss.mse is not None
                and test_loss.psnr_db is not None
                and test_loss.total_rate_bpp is not None
            ), "Either MSE, PSNR db or total rate bpp is None"
            all_mse_losses.append(test_loss.mse)
            all_psnr_db.append(test_loss.psnr_db)
            all_total_rate_bpp.append(test_loss.total_rate_bpp)

        loss_tensor = torch.tensor(all_losses)
        avg_loss = torch.mean(loss_tensor).item()
        std_loss = torch.std(loss_tensor).item()
        avg_mse = torch.mean(torch.tensor(all_mse_losses)).item()
        avg_psnr_db = torch.mean(torch.tensor(all_psnr_db)).item()
        avg_total_rate_bpp = torch.mean(torch.tensor(all_total_rate_bpp)).item()
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
    recipe: PresetConfig,
    lmbda: float,
    workdir: Path,
    device: POSSIBLE_DEVICE,
    unfreeze_backbone_samples: int,
):
    wholenet = wholenet.to(device)
    batch_size = train_data.batch_size
    assert batch_size is not None, "Batch size is None"

    # Preliminary eval, to have a baseline of test loss.
    best_model = wholenet.state_dict()
    prelim_eval = evaluate_wholenet(wholenet, test_data, lmbda=lmbda, device=device)
    best_test_loss = prelim_eval["test_loss"]

    # We cycle through the training data until necessary.
    train_iter = cycle(train_data)
    samples_seen = 0
    batch_n = 0

    # Starting training.
    wholenet.freeze_resnet()
    for phase_num, training_phase in enumerate(recipe.all_phases):
        print(f"Starting phase {phase_num + 1}/{len(recipe.all_phases)}")
        print(training_phase)
        phase_total_it = training_phase.max_itr
        optimizer = torch.optim.Adam(wholenet.parameters(), lr=training_phase.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            phase_total_it,
            eta_min=training_phase.end_lr
            if training_phase.end_lr is not None
            else 1e-6,
        )

        train_losses = RunningTrainLoss()
        for phase_it in range(phase_total_it):
            # Iterate over the training data.
            # When we run out of batches, we start from the beginning.
            img_batch = next(train_iter)
            img_batch = img_batch.to(device)
            cur_softround_t = _linear_schedule(
                *training_phase.softround_temperature,
                samples_seen,
                phase_total_it * batch_size,
            )
            cur_noise_param = _linear_schedule(
                *training_phase.noise_parameter,
                samples_seen,
                phase_total_it * batch_size,
            )
            raw_out, rate, add_data = wholenet.forward(
                img_batch,
                softround_temperature=cur_softround_t,
                noise_parameter=cur_noise_param,
                quantizer_noise_type=training_phase.quantizer_noise_type,
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
            train_losses.add(loss_function_output)
            loss_function_output.loss.backward()

            # Gradient accumulation. Probably not a good idea to use with batches larger than 1.
            if (samples_seen % training_phase.gradient_accumulation) < batch_size:
                # Clip gradients to avoid exploding gradients.
                torch.nn.utils.clip_grad_norm_(wholenet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_n += 1
            samples_seen += batch_size
            scheduler.step()

            eval_results = None  # To make sure we only take fresh eval results.

            # In NO coolchic, logging every 5k samples means roughly every 5 mins.
            if (samples_seen % training_phase.freq_valid) < batch_size:
                # Average train losses.
                # Evaluate on test data
                eval_results = evaluate_wholenet(
                    wholenet, test_data, lmbda=lmbda, device=device
                )
                print(eval_results)
                wandb.log(
                    {
                        "samples_seen": samples_seen,
                        "batch": batch_n,
                        "phase_iteration": phase_it,
                        "n_iterations": samples_seen // batch_size,
                        **train_losses.average(),
                        **eval_results,
                    }
                )
                train_losses = RunningTrainLoss()

            # Patience: save model every `patience` samples if it's better than the best model so far.
            # In NO coolchic we go over 50k samples every hour.
            if (samples_seen % training_phase.patience) < batch_size:
                if eval_results is None:
                    # Evaluate on test data
                    eval_results = evaluate_wholenet(
                        wholenet, test_data, lmbda=lmbda, device=device
                    )
                    print(eval_results)
                # Only save if it's better than the best model so far.
                if eval_results["test_loss"] < best_test_loss:
                    best_model = wholenet.state_dict()
                    best_test_loss = eval_results["test_loss"]
                else:
                    # Reset to last best model.
                    wholenet.load_state_dict(best_model)

            if (samples_seen % training_phase.checkpointing_freq) < batch_size:
                save_path = workdir / f"samples_{samples_seen}.pt"
                torch.save(best_model, save_path)

            # Unfreeze backbone if needed
            if samples_seen > unfreeze_backbone_samples:
                wholenet.unfreeze_resnet()

    return wholenet


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

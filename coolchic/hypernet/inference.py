import argparse
from pathlib import Path
from typing import TypeVar

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from tqdm import tqdm

from coolchic.enc.io.io import load_frame_data_from_tensor
from coolchic.enc.training.loss import LossFunctionOutput, loss_function
from coolchic.enc.training.quantizemodel import quantize_model, quantize_model_deltas
from coolchic.hypernet.hypernet import (
    CoolchicWholeNet,
    DeltaWholeNet,
    NOWholeNet,
    WholeNet,
)
from coolchic.utils.nn import get_rate_from_rate_per_module
from coolchic.utils.paths import DATA_DIR, DATASET_NAME
from coolchic.utils.tensors import load_img_from_path
from coolchic.utils.types import HypernetRunConfig, load_config

LMBDA: float | None = None

T = TypeVar("T", bound=WholeNet)


def load_hypernet(
    weights_path: Path, config: HypernetRunConfig, wholenet_cls: type[T]
) -> T:
    # CoolchicWholeNet, NOWholeNet, or DeltaWholeNet.
    net = wholenet_cls(config=config.hypernet_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # Cold run in the network.
    # In particular, this forward pass removes the latent grids from the NO coolchic network parameters.
    # When loading the weights we won't provide them, so this is good.
    net.forward(torch.zeros(1, 3, 256, 256).to(device))
    net.zero_grad()

    # Loading weights.
    weights = torch.load(weights_path, map_location=device, weights_only=True)

    if isinstance(net, CoolchicWholeNet):
        weights = {k: v for k, v in weights.items() if k in net.hypernet.state_dict()}
        net.hypernet.load_state_dict(state_dict=weights)
    else:
        weights = {k: v for k, v in weights.items() if k in net.state_dict()}
        # Filter out latent grids, they are not relevant for inference.
        weights = {
            k: v
            if "latent_grids" not in k
            else torch.zeros_like(dict(net.named_parameters())[k])
            for k, v in weights.items()
        }
        net.load_state_dict(state_dict=weights)

    return net


def get_image_from_hypernet(
    net: WholeNet, img_path: Path, lmbda: float, mlp_rate: bool
) -> tuple[torch.Tensor, LossFunctionOutput]:
    img = load_img_from_path(img_path)
    img = load_frame_data_from_tensor(img).data
    assert isinstance(img, torch.Tensor)  # To make pyright happy.
    device = next(net.parameters()).device
    img = img.to(device)

    # Forward pass.
    net.eval()
    with torch.no_grad():
        if not mlp_rate:
            # No need to quantize the model, just a normal forward pass.
            out_img, out_rate, _ = net.forward(
                img, quantizer_noise_type="none", quantizer_type="hardround"
            )
            rate_mlp = 0.0
        else:
            if isinstance(net, DeltaWholeNet):
                latents, synth_deltas, arm_deltas = net.hypernet.forward(img)
                all_deltas = {
                    "synthesis": synth_deltas,
                    "arm": arm_deltas,
                }

                # Get a reference for the forward pass without deltas.
                og_out_img, out_rate, _ = net.mean_decoder.forward(
                    latents, quantizer_noise_type="none", quantizer_type="hardround"
                )
                og_loss = loss_function(
                    og_out_img,
                    out_rate,
                    img,
                    lmbda=lmbda,
                    rate_mlp_bit=0.0,
                    compute_logs=True,
                )

                quantized_deltas, rate_per_module = quantize_model_deltas(
                    net.mean_decoder,
                    latents=latents,
                    all_deltas=all_deltas,
                    input_img=img,
                    lmbda=lmbda,
                )
                new_params = net.add_deltas(
                    synth_delta_dict=quantized_deltas["synthesis"],
                    arm_delta_dict=quantized_deltas["arm"],
                    batch_size=1,
                    remove_batch_dim=True,
                )
                cc_enc = net.mean_decoder.as_coolchic(
                    latents=latents, stop_grads=True, new_parameters=new_params
                )
            else:
                # image to coolchic creates a coolchic encoder with the hypernet weights.
                cc_enc = net.image_to_coolchic(img, stop_grads=True)
                cc_enc._store_full_precision_param()
                cc_enc = quantize_model(encoder=cc_enc, input_img=img, lmbda=lmbda)
                rate_per_module = cc_enc.get_network_rate()
            # Rate of all the mlp weights.
            rate_mlp = get_rate_from_rate_per_module(rate_per_module)
            # Get image from the quantized model (should perform slightly worse).
            out_img, out_rate, _ = cc_enc.forward(
                quantizer_noise_type="none", quantizer_type="hardround"
            )

        loss_out = loss_function(
            out_img,
            out_rate,
            img,
            lmbda=lmbda,
            rate_mlp_bit=rate_mlp,
            compute_logs=True,
        )

        # Compare loss with and without quantized deltas.
        # If the loss is worse, we don't use deltas.
        if isinstance(net, DeltaWholeNet) and mlp_rate:
            # NOTE: losses are tensors and og_loss is bound here,
            # so let's make pyright shut up.
            if og_loss.loss.item() < loss_out.loss.item():  # pyright: ignore
                loss_out = og_loss  # pyright: ignore
                out_img = og_out_img  # pyright: ignore

    assert isinstance(loss_out.total_rate_bpp, float)  # To make pyright happy.
    assert isinstance(loss_out.mse, float)  # To make pyright happy.
    return out_img, loss_out


def img_eval(
    img_path: Path,
    model: WholeNet,
    lmbda: float,
    mlp_rate: bool,
    save: bool = False,
) -> tuple[dict[str, str | float], str]:
    out_img, loss_out = get_image_from_hypernet(model, img_path, lmbda, mlp_rate)
    if save:
        save_path = img_path.with_suffix(f".out{img_path.suffix}").parts[-1]
        torchvision.utils.save_image(out_img, save_path)
    return {
        "seq_name": img_path.stem,
        "rate_bpp": loss_out.total_rate_bpp,
        "rate_latent_bpp": loss_out.rate_latent_bpp,
        "rate_nn_bpp": loss_out.rate_nn_bpp,
        "psnr_db": loss_out.psnr_db,
        "mse": loss_out.mse,
    }, save_path if save else None  # pyright: ignore


def eval_on_whole_dataset(
    model: WholeNet, lmbda: float, mlp_rate: bool, dataset: DATASET_NAME
) -> pd.DataFrame:
    res: list[dict] = []
    all_images = list((DATA_DIR / dataset).glob("*.png"))
    for img_path in tqdm(all_images):
        res.append(img_eval(img_path, model, lmbda, mlp_rate=mlp_rate)[0])
    df = pd.DataFrame(res)
    return df


def main_eval(
    weight_paths: list[Path],
    lmbda: float,
    cfg: HypernetRunConfig,
    wholenet_cls: type[WholeNet],
    mlp_rate: bool,
    dataset: DATASET_NAME,
    workdir: Path | None = None,
) -> None:
    """Evaluate a hypernet in a given dataset."""
    # Initial checks.
    if (wholenet_cls == NOWholeNet) and mlp_rate:
        raise ValueError(
            "When evaluating a NO coolchic hypernet, "
            "MLP rate should not be computed."
        )
    dfs = []
    # More than one model path allowed.
    for i, weight_path in enumerate(weight_paths):
        print(f"Loading model {i + 1}/{len(weight_paths)}")
        model = load_hypernet(weight_path, cfg, wholenet_cls)
        df = eval_on_whole_dataset(model, lmbda, mlp_rate=mlp_rate, dataset=dataset)
        df["anchor"] = "hypernet" if len(weight_paths) == 1 else weight_path.stem
        dfs.append(df)

    whole_df = pd.concat(dfs)
    results_name = f"{dataset}_results.csv"
    whole_df.to_csv(workdir / results_name if workdir else results_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a hypernet's performance.")
    # Comma-separated list of weight paths.
    parser.add_argument(
        "--weight_paths",
        type=str,
        required=True,
        help="Comma-separated list of weight paths. "
        "Only one path is allowed when evaluating a single image.",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to the hypernet config."
    )
    parser.add_argument(
        "--hypernet",
        type=str,
        default="full",
        help="Hypernet type. Can be one of ['full', 'delta', 'nocchic'].",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kodak", "clic20-pro-valid"],
        default="kodak",
        help="Dataset to evaluate on. Can be 'kodak' or 'clic20-pro-valid'.",
    )
    parser.add_argument(
        "--no_mlp_rate",
        action="store_true",
        help="If set, the rate of the MLP weights "
        "will not be included in the evaluation numbers.",
    )
    args = parser.parse_args()

    # Validating and processing arguments.
    run_cfg = load_config(args.config, HypernetRunConfig)
    w_paths = [Path(p) for p in args.weight_paths.split(",")]
    assert all(p.exists() for p in w_paths), "One or more weight paths do not exist."

    assert args.hypernet in ["full", "delta", "nocchic"], "Invalid hypernet type."
    wholenet_cls = (
        CoolchicWholeNet
        if args.hypernet == "full"
        else DeltaWholeNet
        if args.hypernet == "delta"
        else NOWholeNet
    )

    assert isinstance(run_cfg.lmbda, float), (
        "Lambda must be a float." f"Got {run_cfg.lmbda}"
    )
    main_eval(
        w_paths,
        run_cfg.lmbda,
        run_cfg,
        wholenet_cls,
        mlp_rate=not args.no_mlp_rate,
        dataset=args.dataset,
    )
    plt.show()

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
from coolchic.enc.training.quantizemodel import quantize_model
from coolchic.eval.hypernet import plot_hypernet_rd
from coolchic.hypernet.hypernet import (
    CoolchicWholeNet,
    DeltaWholeNet,
    NOWholeNet,
    WholeNet,
)
from coolchic.utils.nn import get_mlp_rate
from coolchic.utils.paths import DATA_DIR
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
    net: WholeNet, img_path: Path, lmbda: float
) -> tuple[torch.Tensor, LossFunctionOutput]:
    img = load_img_from_path(img_path)
    img = load_frame_data_from_tensor(img).data
    assert isinstance(img, torch.Tensor)  # To make pyright happy.
    device = next(net.parameters()).device
    img = img.to(device)

    # Forward pass.
    net.eval()
    with torch.no_grad():
        out_img, out_rate, _ = net.forward(
            img, quantizer_noise_type="none", quantizer_type="hardround"
        )

        # getting mlp rate involves "mocking" a model quantization.
        cc_enc = net.image_to_coolchic(img, stop_grads=True)
        cc_enc._store_full_precision_param()
        cc_enc = quantize_model(encoder=cc_enc, input_img=img, lmbda=lmbda)
        rate_mlp = get_mlp_rate(cc_enc)

        loss_out = loss_function(
            out_img, out_rate, img, lmbda=0.0, rate_mlp_bit=rate_mlp, compute_logs=True
        )
    assert isinstance(loss_out.total_rate_bpp, float)  # To make pyright happy.
    assert isinstance(loss_out.mse, float)  # To make pyright happy.
    return out_img, loss_out


def img_eval(
    img_path: Path, model: WholeNet, lmbda: float
) -> tuple[dict[str, str | float], str]:
    out_img, loss_out = get_image_from_hypernet(model, img_path, lmbda)
    save_path = img_path.with_suffix(f".out{img_path.suffix}").parts[-1]
    torchvision.utils.save_image(out_img, save_path)
    return {  # pyright: ignore
        "seq_name": img_path.stem,
        "rate_bpp": loss_out.total_rate_bpp,
        "psnr_db": loss_out.psnr_db,
        "mse": loss_out.mse,
    }, save_path


def eval_on_all_kodak(model: WholeNet, lmbda: float) -> pd.DataFrame:
    res: list[dict] = []
    for i in tqdm(range(1, 25)):
        img_path = DATA_DIR / "kodak" / f"kodim{i:02d}.png"
        res.append(img_eval(img_path, model, lmbda)[0])
    df = pd.DataFrame(res)
    print(df)
    return df


def main_eval(
    weight_paths: list[Path],
    lmbda: float,
    img_num: int | None,
    img_path: Path | None,
    cfg: HypernetRunConfig,
    wholenet_cls: type[WholeNet],
    workdir: Path | None = None,
) -> None:
    """Evaluate a hypernet, either on a single image or on all Kodak images.

    Logic is a bit complicated, to allow for both single and multiple image evaluation, depending on the arguments.
    Can be used by any script to evaluate a hypernet.
    """
    if img_path is not None or img_num is not None:
        assert (
            len(weight_paths) == 1
        ), "Only one weight path is allowed when evaluating a single image."
        model = load_hypernet(weight_paths[0], cfg, wholenet_cls)

        if img_path is not None:
            compressed, loss = get_image_from_hypernet(model, img_path, lmbda=lmbda)
            print(
                f"Rate: {loss.total_rate_bpp:2f} bpp, MSE: {loss.mse:2f}, PSNR: {loss.psnr_db}"
            )
            torchvision.utils.save_image(compressed, img_path.with_suffix(".out.png"))
        else:
            # img_num is not None.
            img_path = DATA_DIR / "kodak" / f"kodim{img_num:02d}.png"
            image_data, save_path = img_eval(img_path, model, lmbda)
            print("Evaluation results:")
            [print(f"{k}: {v}") for k, v in image_data.items()]
            print(f"Saved to {save_path}")
    else:
        # Evaluate on all Kodak images.
        dfs = []
        # More than one model path allowed.
        for i, weight_path in enumerate(weight_paths):
            print(f"Loading model {i + 1}/{len(weight_paths)}")
            model = load_hypernet(weight_path, cfg, wholenet_cls)
            df = eval_on_all_kodak(model, lmbda)
            df["anchor"] = "hypernet" if len(weight_paths) == 1 else weight_path.stem
            dfs.append(df)

        whole_df = pd.concat(dfs)
        whole_df.to_csv(
            workdir / "kodak_results.csv" if workdir else "kodak_results.csv"
        )
        for kodim_name in [f"kodim{i:02d}" for i in range(1, 25)]:
            plot_hypernet_rd(kodim_name, whole_df)


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
        "--img_num",
        type=int,
        default=None,
        help="Evaluate a single Kodak image by number.",
    )
    parser.add_argument(
        "--img_path", type=Path, default=None, help="Evaluate a single image by path."
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
        w_paths, run_cfg.lmbda, args.img_num, args.img_path, run_cfg, wholenet_cls
    )
    plt.show()

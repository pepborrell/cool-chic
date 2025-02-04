import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from tqdm import tqdm

from coolchic.enc.io.io import load_frame_data_from_tensor
from coolchic.enc.training.loss import LossFunctionOutput, loss_function
from coolchic.eval.hypernet import plot_hypernet_rd
from coolchic.hypernet.hypernet import CoolchicWholeNet
from coolchic.utils.paths import DATA_DIR
from coolchic.utils.types import HypernetRunConfig, load_config


def load_hypernet(weights_path: Path, config: HypernetRunConfig) -> CoolchicWholeNet:
    # Loading weights.
    net = CoolchicWholeNet(config=config.hypernet_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    weights = torch.load(weights_path, map_location=device, weights_only=True)

    # Backward compatibility. We used to save the whole net, now we only save the hypernet.
    weights = {k.replace("hypernet.", ""): v for k, v in weights.items()}
    weights = {k: v for k, v in weights.items() if k in net.hypernet.state_dict()}

    net.hypernet.load_state_dict(state_dict=weights)

    return net


def get_image_from_hypernet(
    net: CoolchicWholeNet, img_path: Path
) -> tuple[torch.Tensor, LossFunctionOutput]:
    # Load image.
    img = torchvision.io.decode_image(
        str(img_path), mode=torchvision.io.ImageReadMode.RGB
    )
    img = load_frame_data_from_tensor(img).data
    assert isinstance(img, torch.Tensor)  # To make pyright happy.

    # Forward pass.
    net.eval()
    with torch.no_grad():
        out_img, out_rate, _ = net.forward(
            img, quantizer_noise_type="none", quantizer_type="hardround"
        )
        rate_mlp = net.get_mlp_rate()
        loss_out = loss_function(
            out_img, out_rate, img, lmbda=0.0, rate_mlp_bit=rate_mlp, compute_logs=True
        )
    assert isinstance(loss_out.total_rate_bpp, float)  # To make pyright happy.
    assert isinstance(loss_out.mse, float)  # To make pyright happy.
    return out_img, loss_out


def img_eval(
    img_path: Path, model: CoolchicWholeNet
) -> tuple[dict[str, str | float], str]:
    out_img, loss_out = get_image_from_hypernet(model, img_path)
    save_path = img_path.with_suffix(f".out{img_path.suffix}").parts[-1]
    torchvision.utils.save_image(out_img, save_path)
    return {  # pyright: ignore
        "seq_name": img_path.stem,
        "rate_bpp": loss_out.total_rate_bpp,
        "psnr_db": loss_out.psnr_db,
        "mse": loss_out.mse,
    }, save_path


def eval_on_all_kodak(model: CoolchicWholeNet):
    res: list[dict] = []
    for i in tqdm(range(1, 25)):
        img_path = DATA_DIR / "kodak" / f"kodim{i:02d}.png"
        res.append(img_eval(img_path, model)[0])
    df = pd.DataFrame(res)
    print(df)
    return df


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
    args = parser.parse_args()

    run_cfg = load_config(args.config, HypernetRunConfig)
    w_paths = [Path(p) for p in args.weight_paths.split(",")]
    assert all(p.exists() for p in w_paths), "One or more weight paths do not exist."

    if args.img_path is not None or args.img_num is not None:
        assert (
            len(w_paths) == 1
        ), "Only one weight path is allowed when evaluating a single image."
        model = load_hypernet(w_paths[0], run_cfg)

        if args.img_path is not None:
            compressed, loss = get_image_from_hypernet(model, args.img_path)
            print(
                f"Rate: {loss.total_rate_bpp:2f} bpp, MSE: {loss.mse:2f}, PSNR: {loss.psnr_db}"
            )
            torchvision.utils.save_image(
                compressed, args.img_path.with_suffix(".out.png")
            )
        else:
            # img_num is not None.
            img_path = DATA_DIR / "kodak" / f"kodim{args.img_num:02d}.png"
            image_data, save_path = img_eval(img_path, model)
            print("Evaluation results:")
            [print(f"{k}: {v}") for k, v in image_data.items()]
            print(f"Saved to {save_path}")
    else:
        # Evaluate on all Kodak images.
        dfs = []
        # More than one model path allowed.
        for i, weight_path in enumerate(w_paths):
            print(f"Loading model {i + 1}/{len(w_paths)}")
            model = load_hypernet(weight_path, run_cfg)
            df = eval_on_all_kodak(model)
            df["anchor"] = "hypernet" if len(w_paths) == 1 else weight_path.stem
            dfs.append(df)

        whole_df = pd.concat(dfs)
        whole_df.to_csv("kodak_results.csv")
        for kodim_name in [f"kodim{i:02d}" for i in range(1, 25)]:
            plot_hypernet_rd(kodim_name, whole_df)
        plt.show()

import argparse
from pathlib import Path

import torch
import torchvision
import yaml

from coolchic.enc.io.io import load_frame_data_from_tensor
from coolchic.enc.training.loss import LossFunctionOutput, loss_function
from coolchic.hypernet.hypernet import CoolchicWholeNet
from coolchic.utils.paths import DATA_DIR
from coolchic.utils.types import HyperNetConfig, HypernetRunConfig


def get_image_from_hypernet(
    weights_path: Path, hn_config: HyperNetConfig, img_path: Path
) -> tuple[torch.Tensor, LossFunctionOutput]:
    net = CoolchicWholeNet(config=hn_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    weights = {k.replace("hypernet.", ""): v for k, v in weights.items()}
    weights = {k: v for k, v in weights.items() if k in net.hypernet.state_dict()}
    net.hypernet.load_state_dict(state_dict=weights)
    img = torchvision.io.decode_image(
        str(img_path), mode=torchvision.io.ImageReadMode.RGB
    )
    img = load_frame_data_from_tensor(img).data

    assert isinstance(img, torch.Tensor)  # To make pyright happy.
    out_img, out_rate, _ = net.forward(img)
    loss_out = loss_function(
        out_img, out_rate, img, lmbda=0.0, rate_mlp_bit=0.0, compute_logs=True
    )
    assert isinstance(loss_out.total_rate_bpp, float)  # To make pyright happy.
    assert isinstance(loss_out.mse, float)  # To make pyright happy.
    return out_img, loss_out


def show_kodak(img_num: int, weights_path: Path, hn_config: HyperNetConfig) -> None:
    img_path = DATA_DIR / "kodak" / f"kodim{img_num:02d}.png"
    out_img, loss_out = get_image_from_hypernet(weights_path, hn_config, img_path)
    print(
        f"image: {img_path.name}. "
        f"Rate: {loss_out.total_rate_bpp:2f} bpp, MSE: {loss_out.mse:2f}, PSNR: {loss_out.psnr_db}"
    )
    # Show image
    torchvision.utils.save_image(out_img, f"kodim_{img_num:02d}.out.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=Path)
    parser.add_argument("--img_num", type=int, default=None)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        run_cfg = HypernetRunConfig(**yaml.safe_load(stream))
    if args.img_num is not None:
        show_kodak(args.img_num, args.weights_path, run_cfg.hypernet_cfg)
    else:
        for i in range(1, 25):
            show_kodak(i, args.weights_path, run_cfg.hypernet_cfg)

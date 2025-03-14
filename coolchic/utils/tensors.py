from pathlib import Path

import torch
import torchvision


def load_img_from_path(img_path: Path) -> torch.Tensor:
    # Load image from filesystem to tensor.
    img = torchvision.io.decode_image(
        str(img_path), mode=torchvision.io.ImageReadMode.RGB
    )
    return img

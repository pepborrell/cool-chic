import torch
from torch.utils.data import Dataset

from coolchic.enc.utils.misc import POSSIBLE_DEVICE
from coolchic.metalearning.training_data import get_image_list, image_to_tensor

PATCH_WIDTH = PATCH_HEIGHT = 512
PATCH_SIZE = (PATCH_HEIGHT, PATCH_WIDTH)


class OpenImagesDataset(Dataset):
    def __init__(
        self, n_images: int = 1000, device: POSSIBLE_DEVICE = "cuda:0"
    ) -> None:
        self.n_images = n_images
        self.img_ids = get_image_list(n_images)
        self.device = device

    def __len__(self) -> int:
        return self.n_images

    @staticmethod
    def extract_random_patch(img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        # Set random seed for reproducibility.
        torch.manual_seed(1999)
        i = torch.randint(0, h - PATCH_HEIGHT, (1,)).item()
        j = torch.randint(0, w - PATCH_WIDTH, (1,)).item()
        return img[..., i : i + PATCH_HEIGHT, j : j + PATCH_WIDTH]

    def __getitem__(self, index) -> torch.Tensor:
        img_path = self.img_ids[index]
        img = image_to_tensor(img_path)
        patch = self.extract_random_patch(img)
        return patch.to(self.device)

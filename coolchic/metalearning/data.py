import torch
from torch.utils.data import Dataset

from coolchic.enc.io.io import load_frame_data_from_tensor
from coolchic.enc.utils.misc import POSSIBLE_DEVICE
from coolchic.metalearning.training_data import get_image_list, image_to_tensor

PATCH_WIDTH = PATCH_HEIGHT = 512
PATCH_SIZE = (PATCH_HEIGHT, PATCH_WIDTH)


class OpenImagesDataset(Dataset):
    def __init__(self, n_images: int = 1000, device: POSSIBLE_DEVICE = "cpu") -> None:
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
        if h < PATCH_HEIGHT or w < PATCH_WIDTH:
            # Work with the full image if it is too small.
            return img
        i = torch.randint(0, h - PATCH_HEIGHT, (1,)).item()
        j = torch.randint(0, w - PATCH_WIDTH, (1,)).item()
        return img[..., i : i + PATCH_HEIGHT, j : j + PATCH_WIDTH]

    def _getitem_one(self, index: int) -> torch.Tensor:
        img_path = self.img_ids[index]
        img = image_to_tensor(img_path)
        patch = self.extract_random_patch(img)
        patch_correct = load_frame_data_from_tensor(patch).data
        assert isinstance(patch_correct, torch.Tensor)
        return patch_correct.to(self.device)

    def _getitem_slice(self, indices: slice) -> torch.Tensor:
        patches = []
        start = indices.start if indices.start is not None else 0
        stop = indices.stop if indices.stop is not None else len(self)
        step = indices.step if indices.step is not None else 1
        for i in range(start, stop, step):
            patches.append(self._getitem_one(i))
        return torch.stack(patches)

    def __getitem__(self, index: int | slice) -> torch.Tensor:
        if isinstance(index, int):
            return self._getitem_one(index)
        return self._getitem_slice(index)

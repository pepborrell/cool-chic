import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from coolchic.metalearning.training_data import get_images

PATCH_WIDTH = PATCH_HEIGHT = 256
PATCH_SIZE = (PATCH_HEIGHT, PATCH_WIDTH)


class OpenImagesDataset(Dataset):
    def __init__(self, n_images: int = 100) -> None:
        self.img_paths = get_images(n_images)
        self.full_images = [read_image(str(img_path)) for img_path in self.img_paths]
        self.all_patches = [self.get_patches(i) for i in range(len(self.full_images))]

    def num_patches(self, image: torch.Tensor) -> int:
        _, H, W = image.shape
        new_H = H - H % PATCH_HEIGHT
        new_W = W - W % PATCH_WIDTH
        return (new_H // PATCH_HEIGHT) * (new_W // PATCH_WIDTH)

    def get_patches(self, index) -> torch.Tensor:
        image = self.full_images[index]
        # Discard elements that do not fit the patch size.
        _, H, W = image.shape
        new_H = H - H % PATCH_HEIGHT
        new_W = W - W % PATCH_WIDTH
        image = image[:, :new_H, :new_W]

        # Use unfold to extract patches
        patches = image.unfold(1, PATCH_HEIGHT, PATCH_HEIGHT).unfold(
            2, PATCH_WIDTH, PATCH_WIDTH
        )
        # Rearrange to (num_patches, C, patch_height, patch_width)
        patches = (
            patches.permute(1, 2, 0, 3, 4)
            .contiguous()
            .view(-1, image.size(0), PATCH_HEIGHT, PATCH_WIDTH)
        )
        return patches

    def __len__(self) -> int:
        return len(self.all_patches)

    def __getitem__(self, index) -> torch.Tensor:
        return self.all_patches[index]

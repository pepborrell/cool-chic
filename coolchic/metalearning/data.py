import torch
import torchvision
from torch.utils.data import Dataset

from coolchic.enc.io.io import load_frame_data_from_tensor
from coolchic.metalearning.training_data import get_image_list


class OpenImagesDataset(Dataset):
    def __init__(
        self,
        n_images: int,
        patch_size: tuple[int, int] | None,
        train: bool = True,
        add_batch_dim: bool = False,
    ) -> None:
        self.n_images = n_images
        self.train = train
        n_test = min(64, int(n_images * 0.2))  # At most 64 test images.
        self.n_train = n_images - n_test
        self.img_paths = get_image_list(n_images)
        self.train_paths = self.img_paths[: self.n_train]
        self.test_paths = self.img_paths[self.n_train :]

        self.patch_size = patch_size
        self.add_batch_dim = add_batch_dim

    def __len__(self) -> int:
        return self.n_train if self.train else self.n_images - self.n_train

    @staticmethod
    def extract_random_patch(
        img: torch.Tensor, patch_size: tuple[int, int] | None
    ) -> torch.Tensor:
        if patch_size is None:
            return img
        h, w = img.shape[-2:]
        patch_height, patch_width = patch_size
        if h < patch_height or w < patch_width:
            # Work with the full image if it is too small.
            return img
        # Set random seed for reproducibility. Random seed is based on the image content.
        torch.manual_seed(torch.sum(img).item())
        i = torch.randint(0, h - patch_height, (1,)).item()
        j = torch.randint(0, w - patch_width, (1,)).item()
        return img[..., i : i + patch_height, j : j + patch_width]

    def _getitem_one(self, index: int) -> torch.Tensor:
        img_paths = self.train_paths if self.train else self.test_paths
        img_path = img_paths[index]
        # Load image from filesystem to tensor.
        img = torchvision.io.decode_image(
            str(img_path), mode=torchvision.io.ImageReadMode.RGB
        )
        patch = self.extract_random_patch(img, self.patch_size)
        patch_correct = load_frame_data_from_tensor(patch).data
        assert isinstance(patch_correct, torch.Tensor)

        # load_frame_data_from_tensor returns a tensor with shape (1, 3, 256, 256).
        # Remove the batch dimension if desired, it will be added by the dataloader.
        if self.add_batch_dim:
            return patch_correct
        return patch_correct.squeeze(0)

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

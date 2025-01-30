from pathlib import Path

from coolchic.utils.paths import OPEN_IMAGES_DIR


def get_image_list(n_images: int = 100) -> list[Path]:
    """Returns a list of the first N images downloaded in the openimages directory."""
    img_list = []
    # Images are downloaded into train_1, train_2, ..., train_N directories.
    train_dirs = [
        train_dir
        for train_dir in OPEN_IMAGES_DIR.iterdir()
        if train_dir.is_dir() and "train" in train_dir.name
    ]
    for train_dir in train_dirs:
        for img in train_dir.iterdir():
            if img.is_file():
                img_list.append(img)
            if len(img_list) == n_images:
                break

    assert len(img_list) >= n_images, "Not enough images in the Open Images directory."

    return img_list

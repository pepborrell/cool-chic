import argparse
import os
import random
import re
import sys
from concurrent import futures
from pathlib import Path

import boto3
import botocore
import torch
import torchvision
import tqdm

from coolchic.utils.paths import DATA_DIR

# Based on the code in https://github.com/openimages/dataset/blob/main/downloader.py
BUCKET_NAME = "open-images-dataset"
REGEX = r"(test|train|validation|challenge2018)/([a-fA-F0-9]*)"

# Defining path to download images.
OPENIMAGES_DOWNLOAD_PATH = DATA_DIR / "metalearning" / "openimages"


def check_and_homogenize_one_image(image: str) -> tuple[str, str]:
    split, image_id = re.match(REGEX, image).groups()
    return split, image_id


def check_and_homogenize_image_list(image_list: list[str]) -> list[tuple[str, str]]:
    homog_list = []
    for line_number, image in enumerate(image_list):
        try:
            homog_list.append(check_and_homogenize_one_image(image))
        except (ValueError, AttributeError):
            raise ValueError(
                f"ERROR in line {line_number} of the image list. The following image "
                f'string is not recognized: "{image}".'
            )
    return homog_list


def download_one_image(bucket, split, image_id, download_folder):
    try:
        bucket.download_file(
            f"{split}/{image_id}.jpg", os.path.join(download_folder, f"{image_id}.jpg")
        )
    except botocore.exceptions.ClientError as exception:
        sys.exit(f"ERROR when downloading image `{split}/{image_id}`: {str(exception)}")


def download_all_images(
    download_folder: Path, input_image_list: list[str], num_processes: int = 1
):
    """Downloads all images specified in the input file."""
    bucket = boto3.resource(
        "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    ).Bucket(BUCKET_NAME)

    if not download_folder.exists():
        download_folder.mkdir(parents=True)

    try:
        image_list = check_and_homogenize_image_list(input_image_list)
    except ValueError as exception:
        sys.exit(f"ERROR: {str(exception)}")

    progress_bar = tqdm.tqdm(
        total=len(image_list), desc="Downloading images", leave=True
    )
    with futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        all_futures = [
            executor.submit(
                download_one_image, bucket, split, image_id, download_folder
            )
            for (split, image_id) in image_list
        ]
        for future in futures.as_completed(all_futures):
            future.result()
            progress_bar.update(1)
    progress_bar.close()


def download_image_to_tensor(image_path: str) -> torch.Tensor:
    s3_client = boto3.client(
        "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    )
    response = s3_client.get_object(
        Bucket=BUCKET_NAME,
        Key=f"{image_path}.jpg",
    )
    image_data = response["Body"].read()
    image_tensor = torchvision.io.decode_image(
        torch.tensor(bytearray(image_data), dtype=torch.uint8),
        mode=torchvision.io.ImageReadMode.RGB,
    )
    return image_tensor


def select_images(image_csv: Path, n_images: int = 100) -> list[str]:
    # We select a subset of N images from the first 100N images in the list.
    random.seed(42)  # Images will always be the same.
    cand_pool_size = min(
        100 * n_images, int(1.7 * 10**6)
    )  # 1.7M images in the dataset.
    indices = set(random.sample(range(1, cand_pool_size), n_images))
    selected_lines = []
    with open(image_csv, "r") as file:
        for i, line in enumerate(file):
            if i > cand_pool_size:
                break
            if i in indices:
                selected_lines.append(line)
    assert len(selected_lines) == n_images

    def format_image_str(line):
        # Lines have several fields. The interesting ones are the id (first field), and the subset (second).
        split = line.split(",")[1]
        img_id = line.split(",")[0]
        return f"{split}/{img_id}"

    return [format_image_str(line) for line in selected_lines]


def get_image_list(n_images: int = 100) -> list[str]:
    """Returns a list of the selected images to download from
    the Open Images dataset.
    """
    # Download from https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv
    img_list_path = DATA_DIR / "metalearning" / "train-images-boxable-with-rotation.csv"
    assert img_list_path.exists(), (
        "Training images list not found. "
        "Please download it from "
        "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
    )
    img_list = select_images(
        img_list_path,
        n_images=n_images,
    )
    return img_list


def get_image_save_path(image_name: str) -> Path:
    return OPENIMAGES_DOWNLOAD_PATH / f"{image_name.split('/')[1]}.jpg"


def filter_list_if_downloaded(img_list: list[str]) -> list[str]:
    return [img for img in img_list if not get_image_save_path(img).exists()]


def select_download_all_images(n_images: int = 100) -> list[Path]:
    """Downloads a subset of images from the Open Images dataset."""
    OPENIMAGES_DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    img_list = get_image_list(n_images=n_images)
    img_list = filter_list_if_downloaded(img_list)
    download_all_images(
        download_folder=OPENIMAGES_DOWNLOAD_PATH,
        input_image_list=img_list,
        num_processes=4,
    )
    return [get_image_save_path(img_name) for img_name in img_list]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_images", type=int, required=True, help="Number of images to download."
    )
    args = parser.parse_args()

    select_download_all_images(n_images=args.n_images)

import requests
from zipfile import ZipFile
from pathlib import Path

# URL of the file to download
url = "https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip"

# Destination path
dataset_name = Path(__file__).stem
assert dataset_name == "clic20-pro-valid"

cur_path = Path(__file__)
output_dir = cur_path.parent.parent / dataset_name
zip_path = output_dir / f"{dataset_name}.zip"


def download_file(url: str, output_path: Path):
    """Downloads the file from the given URL."""
    output_path = output_path.absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Send GET request
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Raise HTTPError for bad responses
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    print(f"File downloaded to: {output_path}")


def unzip(path_to_zipfile: Path, output_dir: Path) -> None:
    """Unzips the zipfile to the output directory."""
    with ZipFile(path_to_zipfile) as f:
        f.extractall(output_dir)
    print(f"Files extracted to: {output_dir}.")


if __name__ == "__main__":
    download_file(url, zip_path)
    unzip(zip_path, output_dir)
    zip_path.unlink()

    # Move files from valid/ to the root of the dataset directory
    valid_dir = output_dir / "valid"
    for item in valid_dir.iterdir():
        if item.is_file():
            item.rename(output_dir / item.name)
        else:
            print(f"Skipping non-file item: {item}")
    # Remove the now-empty valid directory
    valid_dir.rmdir()

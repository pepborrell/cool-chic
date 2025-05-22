import datetime
import shutil
from pathlib import Path


def dirs_to_copy(exp_path: Path):
    all_dirs = []

    for subdir in exp_path.iterdir():
        if not subdir.is_dir():
            continue
        for subsubdir in subdir.iterdir():
            # Get the creation time
            creation_time = subsubdir.stat().st_ctime
            creation_date = datetime.datetime.fromtimestamp(creation_time)
            if creation_date > datetime.datetime.fromisoformat("2024-11-24"):
                all_dirs.append(subsubdir)
    return all_dirs


def copy_dir(origin: Path, destination: Path):
    shutil.copytree(origin, destination)


if __name__ == "__main__":
    exp_path = Path("results/exps/2024-11-21/")
    dest_dir = Path("results/exps/2024-11-26/")
    dirs = dirs_to_copy(exp_path)
    for dir in dirs:
        copy_dir(dir, dest_dir / dir.relative_to(exp_path))

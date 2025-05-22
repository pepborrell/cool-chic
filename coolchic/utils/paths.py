from pathlib import Path
from typing import Literal

COOLCHIC_PYTHON_ROOT = Path(__file__).parent.parent.absolute()
COOLCHIC_REPO_ROOT = COOLCHIC_PYTHON_ROOT.parent

DATA_DIR = COOLCHIC_REPO_ROOT / "data"
LOCAL_SCRATCH_DIR = Path("/scratch/jborrell")
RESULTS_DIR = COOLCHIC_REPO_ROOT / "results"
CONFIG_DIR = COOLCHIC_REPO_ROOT / "cfg"
# Openimages stored in scratch in nodes, but in data in local.
OPEN_IMAGES_DIR = None
if (LOCAL_SCRATCH_DIR / "openimages").exists():
    OPEN_IMAGES_DIR = LOCAL_SCRATCH_DIR / "openimages"
elif (DATA_DIR / "openimages").exists():
    OPEN_IMAGES_DIR = DATA_DIR / "openimages"

ANCHOR_NAME = Literal["coolchic", "hm", "jpeg", "c3"]
DATASET_NAME = Literal["kodak", "clic20-pro-valid"]
kodak_results = RESULTS_DIR / "image" / "kodak"
clic20_results = RESULTS_DIR / "image" / "clic20-pro-valid"
ALL_ANCHORS: dict[DATASET_NAME, dict[ANCHOR_NAME, Path]] = {
    "kodak": {
        "coolchic": kodak_results / "results.tsv",
        "hm": kodak_results / "hm.tsv",
        "jpeg": kodak_results / "jpeg.tsv",
    },
    "clic20-pro-valid": {
        "coolchic": clic20_results / "results.tsv",
        "hm": clic20_results / "hm.tsv",
        "jpeg": clic20_results / "jpeg.tsv",
        "c3": clic20_results / "c3.tsv",
    },
}


def get_latest_checkpoint(run_dir: Path) -> Path:
    # Weights formatted like samples_130000.pt, we take the highest sample number.
    checkpoints = [file for file in run_dir.iterdir() if file.suffix == ".pt"]
    return max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))

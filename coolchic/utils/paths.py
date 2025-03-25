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

ANCHOR_NAMES = Literal["coolchic", "hm", "jpeg"]
ALL_ANCHORS: dict[ANCHOR_NAMES, Path] = {
    "coolchic": RESULTS_DIR / "image" / "kodak" / "results.tsv",
    "hm": RESULTS_DIR / "image" / "kodak" / "hm.tsv",
    "jpeg": RESULTS_DIR / "image" / "kodak" / "jpeg.tsv",
}

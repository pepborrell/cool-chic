from pathlib import Path

COOLCHIC_PYTHON_ROOT = Path(__file__).parent.parent.absolute()
COOLCHIC_REPO_ROOT = COOLCHIC_PYTHON_ROOT.parent

DATA_DIR = COOLCHIC_REPO_ROOT / "data"
LOCAL_SCRATCH_DIR = Path("/scratch/jborrell")
RESULTS_DIR = COOLCHIC_REPO_ROOT / "results"
# Openimages stored in scratch in nodes, but in data in local.
OPEN_IMAGES_DIR = None
if (LOCAL_SCRATCH_DIR / "openimages").exists():
    OPEN_IMAGES_DIR = LOCAL_SCRATCH_DIR / "openimages"
elif (DATA_DIR / "openimages").exists():
    OPEN_IMAGES_DIR = DATA_DIR / "openimages"

ALL_ANCHORS = {
    "coolchic": RESULTS_DIR / "image" / "kodak" / "results.tsv",
    "hm": RESULTS_DIR / "image" / "kodak" / "hm.tsv",
    "jpeg": RESULTS_DIR / "image" / "kodak" / "jpeg.tsv",
}

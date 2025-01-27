from pathlib import Path

COOLCHIC_PYTHON_ROOT = Path(__file__).parent.parent.absolute()
COOLCHIC_REPO_ROOT = COOLCHIC_PYTHON_ROOT.parent

DATA_DIR = COOLCHIC_REPO_ROOT / "data"
RESULTS_DIR = COOLCHIC_REPO_ROOT / "results"

ALL_ANCHORS = {
    "coolchic": RESULTS_DIR / "image" / "kodak" / "results.tsv",
    "hm": RESULTS_DIR / "image" / "kodak" / "hm.tsv",
    "jpeg": RESULTS_DIR / "image" / "kodak" / "jpeg.tsv",
}

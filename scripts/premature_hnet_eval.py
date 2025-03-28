import argparse
from pathlib import Path

from coolchic.hypernet.hypernet import NOWholeNet
from coolchic.hypernet.inference import main_eval as hypernet_eval
from coolchic.utils.paths import CONFIG_DIR, RESULTS_DIR
from coolchic.utils.types import HypernetRunConfig, load_config

HNET_CLS = NOWholeNet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_path", type=Path, required=True)
    args = parser.parse_args()
    if not args.sweep_path.exists():
        raise FileNotFoundError(f"Path not found: {args.sweep_path}")

    sweep_path: Path = args.sweep_path

    runs = [
        run
        for run in sweep_path.iterdir()
        if run.is_dir() and run.name.startswith("config_")
    ]

    for run in runs:
        # checkpoints are formatted like epoch_5_batch_2570000.pt. Let's sort by sample number.
        highest_checkpoint = max(
            run.glob("*.pt"), key=lambda p: int(p.stem.split("_")[-1])
        )
        config_path = CONFIG_DIR / run.absolute().relative_to(RESULTS_DIR).with_suffix(
            ".yaml"
        )
        assert config_path.exists(), f"Config file not found: {config_path}"
        config = load_config(config_path, HypernetRunConfig)
        assert isinstance(
            config.lmbda, float
        ), f"Lambda must be float, got {config.lmbda}"

        premature_workdir = run / "premature_eval"
        premature_workdir.mkdir(exist_ok=True)
        hypernet_eval(
            weight_paths=[highest_checkpoint],
            lmbda=config.lmbda,
            img_num=None,
            img_path=None,
            cfg=config,
            wholenet_cls=HNET_CLS,
            workdir=premature_workdir,
        )

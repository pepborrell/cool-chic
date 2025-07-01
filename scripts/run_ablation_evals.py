import argparse
from pathlib import Path

from pydantic import BaseModel

from coolchic.hypernet.hypernet import DeltaWholeNet
from coolchic.hypernet.inference import main_eval as hypernet_eval
from coolchic.utils.paths import CONFIG_DIR, COOLCHIC_REPO_ROOT, RESULTS_DIR
from coolchic.utils.types import HypernetRunConfig, load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_path", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kodak", "clic20-pro-valid"],
        required=True,
        help="Dataset to evaluate on. Can be 'kodak' or 'clic20-pro-valid'.",
    )
    parser.add_argument("--exp_idx", type=int)
    args = parser.parse_args()
    if not args.sweep_path.exists():
        raise FileNotFoundError(f"Path not found: {args.sweep_path}")

    hnet_cls = DeltaWholeNet
    sweep_path: Path = args.sweep_path

    # Precompute all configs, so we can batch modify them later.
    class ConfigCheckpoint(BaseModel):
        config: HypernetRunConfig
        checkpoint: Path

    configs_checkpoints: list[ConfigCheckpoint] = []
    runs = [
        run
        for run in sweep_path.iterdir()
        if run.is_dir() and run.name.startswith("config_")
    ]
    for run in runs:
        # Find more recent checkpoint in the run directory.
        # checkpoints are formatted like samples_2570000.pt. Let's sort by sample number.
        all_models = list(run.glob("*.pt"))
        if len(all_models) == 1:
            # If there's only one model, use it.
            highest_checkpoint = all_models[0]
        else:
            highest_checkpoint = max(
                all_models, key=lambda p: int(p.stem.split("_")[-1])
            )

        # Find config and load it.
        config_path = CONFIG_DIR / run.absolute().relative_to(RESULTS_DIR).with_suffix(
            ".yaml"
        )
        assert config_path.exists(), f"Config file not found: {config_path}"
        config = load_config(config_path, HypernetRunConfig)
        assert isinstance(
            config.lmbda, float
        ), f"Lambda must be float, got {config.lmbda}"
        configs_checkpoints.append(
            ConfigCheckpoint(config=config, checkpoint=highest_checkpoint)
        )

    base_workdir = COOLCHIC_REPO_ROOT / "switch-ablation-exps"

    used_parts = []
    # Write a list of dicts where the eight possible combinations of parts and their usage are stored.
    # Parts are: ['synthesis', 'arm', 'upsampling'], and they can be used in two ways: 'used' or 'not_used'.
    for part in ["synthesis", "arm", "upsampling"]:
        prev_used_parts = used_parts.copy()
        used_parts = []
        if not prev_used_parts:
            # If this is the first part, we start with two options: used and not used.
            used_parts.append({part: True})
            used_parts.append({part: False})
        else:
            # For each previously used part, we create two new entries: one with the current part used and one not used.
            # This way we generate all combinations of parts being used or not.
            for elem in prev_used_parts:
                for active in [True, False]:
                    new_elem = elem.copy()
                    new_elem[part] = active
                    used_parts.append(new_elem)

    def get_name_from_parts(parts: dict) -> str:
        """Generate a name based on the parts used."""
        active_parts = [part for part, used in parts.items() if used]
        return "_".join(active_parts) if active_parts else "none"

    selected_exps = (
        [args.exp_idx] if args.exp_idx is not None else list(range(len(used_parts)))
    )

    for exp in selected_exps:
        # Modify config activating and deactivating parts.
        parts = used_parts[exp]
        name = get_name_from_parts(parts)
        print(f"Running experiment {exp} with parts: {name}")
        for cfgcpt in configs_checkpoints:
            cfg = cfgcpt.config
            for key, value in parts.items():
                if key == "synthesis":
                    cfg.hypernet_cfg.synthesis.use_this_part = value
                elif key == "arm":
                    cfg.hypernet_cfg.arm.use_this_part = value
                elif key == "upsampling":
                    cfg.hypernet_cfg.upsampling.use_this_part = value
                else:
                    raise ValueError(f"Unknown part: {key}")

            hypernet_eval(
                weight_paths=[cfgcpt.checkpoint],
                lmbda=cfg.lmbda,
                cfg=cfg,
                wholenet_cls=hnet_cls,
                workdir=base_workdir / name,
                mlp_rate=True if not args.no_mlp_rate else False,
                dataset=args.dataset,
            )

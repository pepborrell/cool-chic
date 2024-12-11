import argparse
import os
from pathlib import Path

import torch
import yaml
from enc.utils.misc import get_best_device
from utils.paths import COOLCHIC_REPO_ROOT
from utils.types import UserConfig

import wandb
from coolchic.enc.component.video import load_video_encoder
from coolchic.utils.types import RunConfig


def train_only_latents(path_encoder: Path, config: RunConfig, workdir: Path):
    # structure:
    # * load the model
    # * replace with random latents
    # * retrain the model

    # loading model
    video_encoder = load_video_encoder(path_encoder)

    # replace with random latents
    # we first get the frame
    frame = video_encoder.coding_structure.get_frame_from_coding_order(0)
    assert frame is not None
    # this needs to be False for the optimization to happen.
    frame.already_encoded = False
    frame_encoder, _ = video_encoder.all_frame_encoders[str(frame.coding_order)]
    encoder = frame_encoder.coolchic_encoder
    # reinitialize the latents (to 0 values).
    encoder.initialize_latent_grids()

    # retrain the model.
    # Automatic device detection
    device = get_best_device()
    print(f'{"Device":<20}: {device}')

    if device == "cuda:0":
        torch.backends.cudnn.benchmark = True

    os.environ["WANDB_MODE"] = "online"
    # Start wandb logging.
    wandb.init(project="coolchic-runs", config=config.model_dump())

    _ = video_encoder.encode(
        path_original_sequence=str(config.input.absolute()),
        device=device,
        workdir=workdir,
        job_duration_min=-1,
    )

    video_encoder_savepath = workdir / "video_encoder.pt"
    video_encoder.save(video_encoder_savepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("source_workdir", type=Path)
    args = parser.parse_args()

    assert args.config.exists(), f"Config file {args.config} does not exist."
    assert (
        args.source_workdir.exists()
    ), f"Source workdir {args.source_workdir} does not exist."

    with open(args.config, "r") as stream:
        user_config = UserConfig(**yaml.safe_load(stream))

    # One user config generates one or more runs, depending on the parameters specified.
    all_run_configs = user_config.get_run_configs()
    assert len(all_run_configs) == 1, "Only one run is supported."
    config = all_run_configs[0]

    dest_workdir = (
        config.workdir
        if config.workdir is not None
        # If no workdir is specified, results will be saved in results/{path_to_config_relative_to_cfg}/
        else COOLCHIC_REPO_ROOT
        / "results"
        / args.config.relative_to("cfg").with_suffix("")
        / config.unique_id  # unique id to distinguish different runs launched by the same config.
    )
    dest_workdir.mkdir(parents=True, exist_ok=True)

    source_workdir = Path("results/exps/...")
    path_video_encoder = source_workdir / "video_encoder.pt"
    train_only_latents(path_video_encoder, config, dest_workdir)

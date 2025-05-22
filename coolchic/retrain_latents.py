import argparse
import copy
import os
from pathlib import Path

import wandb
import yaml
from enc.component.coolchic import CoolChicEncoderParameter
from enc.component.video import VideoEncoder, load_video_encoder
from enc.io.io import load_frame_data_from_file
from enc.training.test import test
from enc.training.train import train
from enc.utils.codingstructure import CodingStructure
from enc.utils.manager import FrameEncoderManager
from enc.utils.misc import get_best_device
from enc.utils.parsecli import (
    get_coding_structure_from_args,
    get_coolchic_param_from_args,
    get_manager_from_args,
)
from utils.get_best_models import get_best_model
from utils.paths import COOLCHIC_REPO_ROOT
from utils.types import RunConfig, UserConfig, load_config


def extract_image_encoder(video_encoder: VideoEncoder):
    frame = video_encoder.coding_structure.get_frame_from_coding_order(0)
    assert frame is not None
    frame_encoder, frame_encoder_manager = video_encoder.all_frame_encoders[
        str(frame.coding_order)
    ]
    return frame_encoder, frame_encoder_manager


def train_only_latents(path_encoder: Path, config: RunConfig, workdir: Path):
    # structure:
    # * load the model
    # * replace with random latents
    # * retrain the model

    # loading model
    print(f"Loading model from {path_encoder}")
    old_vid_encoder = load_video_encoder(path_encoder)
    old_frame_encoder, _ = extract_image_encoder(old_vid_encoder)
    print("Model loaded.")

    # Dump raw parameters into a text file to keep track of.
    with open(workdir / "param.txt", "w") as f_out:
        f_out.write(yaml.dump(config.model_dump()))

    if config.disable_wandb:
        # To disable wandb completely.
        os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_MODE"] = "online"
    # Start wandb logging.
    wandb.init(project="coolchic-runs", config=config.model_dump())

    print("Starting training.")
    # Since the video encoder is trying to do everything for me and it's annoying,
    # i'll operate with images as if i was the video master.
    # First: initialise.
    coding_structure = CodingStructure(**get_coding_structure_from_args(config))
    frame_encoder_manager = FrameEncoderManager(**get_manager_from_args(config))
    coolchic_encoder_parameter = CoolChicEncoderParameter(
        **get_coolchic_param_from_args(config.dec_cfg)
    )
    video_encoder = VideoEncoder(
        coding_structure=coding_structure,
        shared_coolchic_parameter=coolchic_encoder_parameter,
        shared_frame_encoder_manager=frame_encoder_manager,
    )
    frame = coding_structure.get_frame_from_coding_order(0)
    assert frame is not None

    # Loading the frame data is very important, apparently.
    frame.data = load_frame_data_from_file(
        str(config.input.absolute()), frame.display_order
    )
    frame.refs_data = video_encoder.get_ref_data(frame)

    # Reset the latent grids to all zeros.
    if config.user_tag == "random-latents-noise":
        old_frame_encoder.coolchic_encoder.initialize_latent_grids(
            zeros=False, random_seed=1234
        )
    else:
        old_frame_encoder.coolchic_encoder.initialize_latent_grids()

    # Automatic device detection
    device = get_best_device()
    frame.to_device(device)
    old_frame_encoder.to_device(device)

    frame_enc = copy.deepcopy(old_frame_encoder)
    for training_phase in frame_encoder_manager.preset.all_phases:
        # Launch training.
        frame_enc = train(
            frame_encoder=frame_enc,
            frame=frame,
            frame_encoder_manager=frame_encoder_manager,
            start_lr=training_phase.lr,
            end_lr=training_phase.end_lr if training_phase.end_lr is not None else 1e-5,
            cosine_scheduling_lr=training_phase.schedule_lr,
            max_iterations=training_phase.max_itr,
            frequency_validation=training_phase.freq_valid,
            patience=training_phase.patience,
            optimized_module=training_phase.optimized_module,
            quantizer_type=training_phase.quantizer_type,
            quantizer_noise_type=training_phase.quantizer_noise_type,
            softround_temperature=training_phase.softround_temperature,
            noise_parameter=training_phase.noise_parameter,
        )

    # Save results.
    results = test(
        frame_enc,
        frame,
        frame_encoder_manager,
    )
    frame_workdir = workdir / "frame_000"
    with open(f"{frame_workdir}results_best.tsv", "w") as f_out:
        f_out.write(results.pretty_string(show_col_name=True, mode="all") + "\n")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    assert args.config.exists(), f"Config file {args.config} does not exist."
    user_config = load_config(args.config, UserConfig)

    # One user config generates one or more runs, depending on the parameters specified.
    all_run_configs = user_config.get_run_configs()
    for config in all_run_configs:
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

        input_img = config.input.stem
        lambda_value = config.lmbda
        # Gets best model according to the input image and lambda value.
        source_workdir = get_best_model()[(input_img, lambda_value)]
        assert (
            source_workdir.exists()
        ), f"Source workdir {source_workdir} does not exist."

        path_video_encoder = source_workdir / "video_encoder.pt"
        train_only_latents(path_video_encoder, config, dest_workdir)

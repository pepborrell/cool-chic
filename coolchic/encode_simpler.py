import argparse
import copy
import os
from pathlib import Path

import torch
import yaml

import wandb
from coolchic.enc.component.coolchic import CoolChicEncoderParameter
from coolchic.enc.component.video import (
    FrameEncoder,
    FrameEncoderManager,
    VideoEncoder,
    load_video_encoder,
)
from coolchic.enc.io.io import load_frame_data_from_file
from coolchic.enc.training.test import test
from coolchic.enc.training.train import train
from coolchic.enc.training.warmup import warmup
from coolchic.enc.utils.codingstructure import CodingStructure, Frame
from coolchic.enc.utils.misc import get_best_device
from coolchic.enc.utils.parsecli import (
    get_coding_structure_from_args,
    get_coolchic_param_from_args,
    get_manager_from_args,
)
from coolchic.utils.paths import COOLCHIC_REPO_ROOT
from coolchic.utils.types import RunConfig, UserConfig

"""
This file has been simplified to only train one image and remove most complexity introduced by the VideoEncoder class.
"""


def extract_image_encoder(video_encoder: VideoEncoder):
    frame = video_encoder.coding_structure.get_frame_from_coding_order(0)
    assert frame is not None
    frame_encoder, frame_encoder_manager = video_encoder.all_frame_encoders[
        str(frame.coding_order)
    ]
    return frame_encoder, frame_encoder_manager


def build_frame_encoder(param: CoolChicEncoderParameter, frame: Frame):
    assert (
        frame.data is not None
    ), "Frame data must be loaded before building the frame encoder."
    return FrameEncoder(
        coolchic_encoder_param=param,
        frame_type=frame.frame_type,
        frame_data_type=frame.data.frame_data_type,
        bitdepth=frame.data.bitdepth,
    )


def get_workdir(config: RunConfig) -> Path:
    workdir = (
        config.workdir
        if config.workdir is not None
        # If no workdir is specified, results will be saved in results/{path_to_config_relative_to_cfg}/
        else COOLCHIC_REPO_ROOT
        / "results"
        / config_path.relative_to("cfg").with_suffix("")
        / config.unique_id  # unique id to distinguish different runs launched by the same config.
    )
    workdir.mkdir(parents=True, exist_ok=True)
    assert workdir.is_dir()
    return workdir


def start_training(workdir: Path, config: RunConfig) -> None:
    print(
        "\n\n"
        "*----------------------------------------------------------------------------------------------------------*\n"
        "|                                                                                                          |\n"
        "|                                                                                                          |\n"
        "|       ,gggg,                                                                                             |\n"
        '|     ,88"""Y8b,                           ,dPYb,                             ,dPYb,                       |\n'
        "|    d8\"     `Y8                           IP'`Yb                             IP'`Yb                       |\n"
        "|   d8'   8b  d8                           I8  8I                             I8  8I      gg               |\n"
        "|  ,8I    \"Y88P'                           I8  8'                             I8  8'      \"\"               |\n"
        "|  I8'             ,ggggg,      ,ggggg,    I8 dP      aaaaaaaa        ,gggg,  I8 dPgg,    gg     ,gggg,    |\n"
        '|  d8             dP"  "Y8ggg  dP"  "Y8ggg I8dP       """"""""       dP"  "Yb I8dP" "8I   88    dP"  "Yb   |\n'
        "|  Y8,           i8'    ,8I   i8'    ,8I   I8P                      i8'       I8P    I8   88   i8'         |\n"
        "|  `Yba,,_____, ,d8,   ,d8'  ,d8,   ,d8'  ,d8b,_                   ,d8,_    _,d8     I8,_,88,_,d8,_    _   |\n"
        '|    `"Y8888888 P"Y8888P"    P"Y8888P"    8P\'"Y88                  P""Y8888PP88P     `Y88P""Y8P""Y8888PP   |\n'
        "|                                                                                                          |\n"
        "|                                                                                                          |\n"
        "| version 3.4, Nov. 2024                                                                © 2023-2024 Orange |\n"
        "*----------------------------------------------------------------------------------------------------------*\n"
    )

    workdir.mkdir(exist_ok=True)
    # Dump raw parameters into a text file to keep track of them.
    with open(workdir / "param.txt", "w") as f_out:
        f_out.write(yaml.dump(config.model_dump()))


if __name__ == "__main__":
    # =========================== Parse arguments =========================== #
    # By increasing priority order, the arguments work as follows:
    #
    #   1. Default value of --dummy_arg=42 is the base value.
    #
    #   2. If dummy_arg is present in either the encoder configuration file
    #     (--enc_cfg) or the decoder configuration file (--dec_cfg), then it
    #     overrides the default value.
    #
    #   3. If --dummy_arg is explicitly provided in the command line, then it
    #      overrides both the default value and the value listed in the
    #      configuration file.

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Specifies the path to the config file that will be used."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as stream:
        user_config = UserConfig(**yaml.safe_load(stream))

    # One user config generates one or more runs, depending on the parameters specified.
    for config in user_config.get_run_configs():
        workdir = get_workdir(config)
        path_video_encoder = workdir / "video_encoder.pt"

        if config.load_models and os.path.exists(path_video_encoder):
            video_encoder = load_video_encoder(path_video_encoder)
            frame_encoder, frame_encoder_manager = extract_image_encoder(video_encoder)
            coolchic_encoder_parameter = video_encoder.shared_coolchic_parameter

            frame = video_encoder.coding_structure.get_frame_from_coding_order(0)
            assert frame is not None
            assert frame.data is not None
        else:
            start_training(workdir, config)

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
            coolchic_encoder_parameter.set_image_size(frame.data.img_size)
            # Usually, the video encoder adapts the decoder structure to the number of channels of the output.
            # In our case, we only encode images (I frames), so the number of channels is always 3.
            coolchic_encoder_parameter.layers_synthesis = [
                lay.replace("X", str(3))
                for lay in coolchic_encoder_parameter.layers_synthesis
            ]
            frame_encoder = build_frame_encoder(coolchic_encoder_parameter, frame)

            # We need the video encoder for this one operation.
            # TODO: can we do this without the video encoder and drop it completely?
            frame.refs_data = video_encoder.get_ref_data(frame)

        # Automatic device detection
        device = get_best_device()
        print(f'{"Device":<20}: {device}')
        # This makes training faster.
        if "cuda" in device:
            torch.backends.cudnn.benchmark = True
        frame.to_device(device)

        ##### LOGGING #####
        # Setting up all logging using wandb.
        if config.disable_wandb:
            # To disable wandb completely.
            os.environ["WANDB_MODE"] = "disabled"
        else:
            os.environ["WANDB_MODE"] = "online"
        # Start wandb logging.
        wandb.init(project="coolchic-runs", config=config.model_dump())

        ##### WARMUP #####
        # Get the number of candidates from the initial warm-up phase
        n_initial_warmup_candidate = frame_encoder_manager.preset.warmup.phases[
            0
        ].candidates

        list_candidates = []
        torch.set_float32_matmul_precision("high")
        for _ in range(n_initial_warmup_candidate):
            cur_frame_encoder = FrameEncoder(
                coolchic_encoder_param=coolchic_encoder_parameter,
                frame_type=frame.frame_type,
                frame_data_type=frame.data.frame_data_type,
                bitdepth=frame.data.bitdepth,
            ).to(device)
            list_candidates.append(cur_frame_encoder)

        # Show the encoder structure.
        print(list_candidates[0].coolchic_encoder.pretty_string() + "\n\n")

        # Use warm-up to find the best initialization among the list
        # of candidates parameters.
        frame_encoder = warmup(
            frame_encoder_manager=frame_encoder_manager,
            list_candidates=list_candidates,
            frame=frame,
            device=device,
        )
        frame_encoder.to_device(device)

        ##### COMPILE #####
        # When compiling the frame encoder, training goes faster.
        # Compile only after the warm-up to compile only once.
        if frame_encoder_manager.preset.preset_name == "debug":
            print("Skip compilation when debugging")
        else:
            frame_encoder = torch.compile(
                frame_encoder,
                dynamic=False,
                mode="reduce-overhead",
                # Some part of the frame_encoder forward (420-related stuff)
                # are not (yet) compatible with compilation. So we can't
                # capture the full graph for yuv420 frame
                fullgraph=frame.data.frame_data_type != "yuv420",
            )

        # Launch training.
        frame_enc = frame_encoder
        for training_phase in frame_encoder_manager.preset.all_phases:
            frame_enc = train(
                frame_encoder=frame_enc,
                frame=frame,
                frame_encoder_manager=frame_encoder_manager,
                start_lr=training_phase.lr,
                end_lr=training_phase.end_lr
                if training_phase.end_lr is not None
                else 1e-5,
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

        # Saving the trained model in video encoder form.
        video_encoder_savepath = workdir / "video_encoder.pt"
        trained_vid_enc = copy.deepcopy(video_encoder)
        # This is the way trained frame encoders are saved inside of the video encoder.
        trained_vid_enc.all_frame_encoders[str(frame.coding_order)] = (
            copy.deepcopy(frame_enc),
            copy.deepcopy(frame_encoder_manager),
        )
        trained_vid_enc.save(video_encoder_savepath)

        # Omitted the bitstream encoding part for brevity.

        wandb.finish()

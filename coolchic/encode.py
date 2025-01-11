# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import argparse
import os
from pathlib import Path

import yaml
from utils.paths import COOLCHIC_REPO_ROOT
from utils.types import UserConfig

import wandb
from coolchic.enc.component.coolchic import CoolChicEncoderParameter
from coolchic.enc.component.video import (
    FrameEncoderManager,
    VideoEncoder,
    load_video_encoder,
)
from coolchic.enc.utils.codingstructure import CodingStructure
from coolchic.enc.utils.misc import TrainingExitCode, get_best_device
from coolchic.enc.utils.parsecli import (
    get_coding_structure_from_args,
    get_coolchic_param_from_args,
    get_manager_from_args,
)

"""
Use this file to train i.e. encode a GOP i.e. something which starts with one
intra frame and is then followed by <intra_period> inter frames. Note that an
image is simply a GOP of size 1 with no inter frames.
"""

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
    parser.add_argument(
        "--openimages_id",
        help="Which image from openimages to train with",
        required=False,
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as stream:
        user_config = UserConfig(**yaml.safe_load(stream))

    if args.openimages_id is not None:
        assert isinstance(user_config.input, list)
        assert user_config.input[0] == Path(
            "openimages"
        ), "If openimages_id is provided, input must be openimages."

    # One user config generates one or more runs, depending on the parameters specified.
    all_run_configs = user_config.get_run_configs()
    for config in all_run_configs:
        workdir = (
            config.workdir
            if config.workdir is not None
            # If no workdir is specified, results will be saved in results/{path_to_config_relative_to_cfg}/
            else COOLCHIC_REPO_ROOT
            / "results"
            / config_path.relative_to("cfg").with_suffix("")
            / config.unique_id  # unique id to distinguish different runs launched by the same config.
        )
        if str(config.input) == "openimages":
            assert config.workdir is None, "Workdir must be None when using openimages."
            workdir = workdir.parent / args.openimages_id
        workdir.mkdir(parents=True, exist_ok=True)
        assert workdir.is_dir()

        path_video_encoder = workdir / "video_encoder.pt"
        if config.load_models and os.path.exists(path_video_encoder):
            video_encoder = load_video_encoder(path_video_encoder)

        else:
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

            # Dump raw parameters into a text file to keep track
            with open(workdir / "param.txt", "w") as f_out:
                f_out.write(yaml.dump(config.model_dump()))
                # f_out.write(str(args))
                # f_out.write("\n")
                # f_out.write("----------\n")
                # f_out.write(
                #     parser.format_values()
                # )  # useful for logging where different settings came from

            # ----- Parse arguments & construct video encoder
            coding_structure = CodingStructure(**get_coding_structure_from_args(config))
            coolchic_encoder_parameter = CoolChicEncoderParameter(
                **get_coolchic_param_from_args(config.dec_cfg)
            )
            frame_encoder_manager = FrameEncoderManager(**get_manager_from_args(config))

            video_encoder = VideoEncoder(
                coding_structure=coding_structure,
                shared_coolchic_parameter=coolchic_encoder_parameter,
                shared_frame_encoder_manager=frame_encoder_manager,
            )

        # Automatic device detection
        device = get_best_device()
        print(f'{"Device":<20}: {device}')

        print(f"\n{video_encoder.coding_structure.pretty_string()}\n")

        if config.disable_wandb:
            # To disable wandb completely.
            os.environ["WANDB_MODE"] = "disabled"
        else:
            os.environ["WANDB_MODE"] = "online"
        # Start wandb logging.
        wandb.init(project="coolchic-runs", config=config.model_dump())

        exit_code = video_encoder.encode(
            path_original_sequence=str(config.input.absolute()),
            device=device,
            workdir=workdir,
            job_duration_min=config.job_duration_min,
            openimages_id=int(args.openimages_id)
            if args.openimages_id is not None
            else None,
        )

        video_encoder_savepath = workdir / "video_encoder.pt"
        video_encoder.save(video_encoder_savepath)

        # Bitstream
        if (
            config.output is not None
            and config.output != ""
            and exit_code == TrainingExitCode.END
        ):
            from enc.bitstream.encode import encode_video

            config.output.parent.mkdir(parents=True, exist_ok=True)

            encode_video(video_encoder, config.output, hls_sig_blksize=16)

            # For the sake of completeness, we add the encoded video to the workdir too.
            import shutil

            shutil.copy(config.output, workdir / config.output.name)

        wandb.finish()

        # sys.exit(exit_code.value)

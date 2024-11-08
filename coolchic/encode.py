# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import argparse
import os
import sys
from pathlib import Path

from coolchic.utils.paths import COOLCHIC_REPO_ROOT
from coolchic.utils.types import Config
import torch
import yaml
from coolchic.enc.component.coolchic import CoolChicEncoderParameter
from coolchic.enc.component.video import (
    FrameEncoderManager,
    VideoEncoder,
    load_video_encoder,
)
from coolchic.enc.utils.codingstructure import CodingStructure
from coolchic.enc.utils.misc import get_best_device

import wandb

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
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as stream:
        config = Config(**yaml.safe_load(stream))

    workdir = (
        config.workdir
        if config.workdir is not None
        # If no workdir is specified, results will be saved in results/{path_to_config_relative_to_cfg}/
        else COOLCHIC_REPO_ROOT
        / "results"
        / config_path.relative_to("cfg").with_suffix("")
    )
    assert workdir.is_dir()
    workdir.mkdir(parents=True, exist_ok=True)

    path_video_encoder = workdir / "video_encoder.pt"
    if config.load_models and os.path.exists(path_video_encoder):
        video_encoder = load_video_encoder(path_video_encoder)

    else:
        start_print = (
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
            "| version 3.3                                                                           © 2023-2024 Orange |\n"
            "*----------------------------------------------------------------------------------------------------------*\n"
        )

        print(start_print)

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

        # ----- Create coding configuration
        assert (
            config.enc_cfg.intra_period >= 0 and config.enc_cfg.intra_period <= 255
        ), (
            f"Intra period should be "
            f"  in [0, 255]. Found {config.enc_cfg.intra_period}"
        )

        assert config.enc_cfg.p_period >= 0 and config.enc_cfg.p_period <= 255, (
            f"P period should be " f"  in [0, 255]. Found {config.enc_cfg.p_period}"
        )

        input_path_str = str(config.input.absolute())
        is_image = (
            input_path_str.endswith(".png")
            or input_path_str.endswith(".PNG")
            or input_path_str.endswith(".jpeg")
            or input_path_str.endswith(".JPEG")
            or input_path_str.endswith(".jpg")
            or input_path_str.endswith(".JPG")
        )

        if is_image:
            assert config.enc_cfg.intra_period == 0 and config.enc_cfg.p_period == 0, (
                f"Encoding a PNG or JPEG image {config.input} must be done with "
                "intra_period = 0 and p_period = 0. Found intra_period = "
                f"{config.enc_cfg.intra_period} and p_period = {config.enc_cfg.p_period}"
            )

        coding_config = CodingStructure(
            intra_period=config.enc_cfg.intra_period,
            p_period=config.enc_cfg.p_period,
            seq_name=os.path.basename(config.input).split(".")[0],
        )

        # Parse arguments
        layers_synthesis = [
            x for x in config.dec_cfg.layers_synthesis.split(",") if x != ""
        ]
        n_ft_per_res = [
            int(x) for x in config.dec_cfg.n_ft_per_res.split(",") if x != ""
        ]

        assert set(n_ft_per_res) == {
            1
        }, f"--n_ft_per_res should only contains 1. Found {config.dec_cfg.n_ft_per_res}"

        assert len(config.dec_cfg.arm.split(",")) == 2, (
            f"--arm format should be X,Y." f" Found {config.dec_cfg.arm}"
        )

        dim_arm, n_hidden_layers_arm = [int(x) for x in config.dec_cfg.arm.split(",")]

        coolchic_encoder_parameter = CoolChicEncoderParameter(
            layers_synthesis=layers_synthesis,
            dim_arm=dim_arm,
            n_hidden_layers_arm=n_hidden_layers_arm,
            n_ft_per_res=n_ft_per_res,
            upsampling_kernel_size=config.dec_cfg.upsampling_kernel_size,
            static_upsampling_kernel=config.dec_cfg.static_upsampling_kernel,
        )

        frame_encoder_manager = FrameEncoderManager(
            preset_config=config.enc_cfg.recipe,
            start_lr=config.enc_cfg.start_lr,
            lmbda=config.lmbda,
            n_loops=config.enc_cfg.n_train_loops,
            n_itr=config.enc_cfg.n_itr,
        )

        video_encoder = VideoEncoder(
            coding_structure=coding_config,
            shared_coolchic_parameter=coolchic_encoder_parameter,
            shared_frame_encoder_manager=frame_encoder_manager,
        )

    # Automatic device detection
    device = get_best_device()
    print(f'{"Device":<20}: {device}')

    # # ====================== Torchscript JIT parameters ===================== #
    # # From https://github.com/pytorch/pytorch/issues/52286
    # # This is no longer the case with the with torch.jit.fuser
    # # ! This gives a significant (+25 %) speed up
    # torch._C._jit_set_profiling_executor(False)
    # torch._C._jit_set_texpr_fuser_enabled(False)
    # torch._C._jit_set_profiling_mode(False)

    # torch.set_float32_matmul_precision("high")
    # # ====================== Torchscript JIT parameters ===================== #

    if device == "cpu":
        # the number of cores is adjusted wrt to the slurm variable if exists
        n_cores = os.getenv("SLURM_JOB_CPUS_PER_NODE")
        # otherwise use the machine cpu count
        if n_cores is None:
            n_cores = os.cpu_count()

        assert isinstance(
            n_cores, int
        ), "The 'SLURM_JOB_CPUS_PER_NODE' environment variable returned a non-integer value."
        n_cores = int(n_cores)
        print(f'{"CPU cores":<20}: {n_cores}')

    elif device == "cuda:0":
        # ! This one makes the training way faster!
        torch.backends.cudnn.benchmark = True

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
    )

    video_encoder_savepath = workdir / "video_encoder.pt"
    video_encoder.save(video_encoder_savepath)

    # Bitstream
    if config.output != "":
        from coolchic.enc.bitstream.encode import encode_video

        video_encoder = load_video_encoder(video_encoder_savepath)
        encode_video(video_encoder, config.output, hls_sig_blksize=16)

    wandb.finish()

    sys.exit(exit_code.value)

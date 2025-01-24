from pathlib import Path

import torch
from torchvision.extension import os

import wandb
from coolchic.enc.component.coolchic import CoolChicEncoder, CoolChicEncoderParameter
from coolchic.enc.component.video import VideoEncoder
from coolchic.enc.io.io import load_frame_data_from_file
from coolchic.enc.training.presets import TrainerPhase, Warmup
from coolchic.enc.training.train import train as coolchic_train
from coolchic.enc.utils.codingstructure import CodingStructure
from coolchic.enc.utils.manager import FrameEncoderManager
from coolchic.enc.utils.misc import get_best_device
from coolchic.enc.utils.parsecli import (
    get_coolchic_param_from_args,
)
from coolchic.encode_simpler import build_frame_encoder
from coolchic.hypernet.inference import load_hypernet
from coolchic.utils.paths import DATA_DIR
from coolchic.utils.types import HypernetRunConfig, PresetConfig, load_config


def main(img_num: int, preset_config: PresetConfig, from_scratch: bool = False) -> None:
    # 1: Get image
    img_path = DATA_DIR / "kodak" / f"kodim{img_num:02d}.png"
    frame_data = load_frame_data_from_file(str(img_path), 0)
    img = frame_data.data
    assert isinstance(img, torch.Tensor)  # To make pyright happy.

    # 2: Get coolchic representation from hypernet
    weights_path = Path("epoch_54_it_520.pt")
    config_path = Path("cfg/exps/basic_hn/100k.yaml")
    config = load_config(config_path, HypernetRunConfig)
    hnet = load_hypernet(weights_path, config)
    cc_encoder = hnet.image_to_coolchic(img, stop_grads=True)
    hnet.zero_grad()

    # 3: Fully fledged coolchic representation to go to training
    coding_structure = CodingStructure(intra_period=0)
    frame_encoder_manager = FrameEncoderManager(
        preset_config=preset_config,
        lmbda=config.lmbda,
    )
    coolchic_encoder_parameter = CoolChicEncoderParameter(
        **get_coolchic_param_from_args(config.hypernet_cfg.dec_cfg)
    )
    video_encoder = VideoEncoder(
        coding_structure=coding_structure,
        shared_coolchic_parameter=coolchic_encoder_parameter,
        shared_frame_encoder_manager=frame_encoder_manager,
    )
    frame = coding_structure.get_frame_from_coding_order(0)
    assert frame is not None  # To make pyright happy.
    frame.data = frame_data
    coolchic_encoder_parameter.set_image_size(frame.data.img_size)
    frame_enc = build_frame_encoder(coolchic_encoder_parameter, frame)

    if from_scratch:
        cc_encoder = CoolChicEncoder(coolchic_encoder_parameter)
    frame_enc.coolchic_encoder = cc_encoder

    # We need the video encoder for this one operation.
    # TODO: can we do this without the video encoder and drop it completely?
    frame.refs_data = video_encoder.get_ref_data(frame)

    # 4: Train like in coolchic
    # Automatic device detection
    device = get_best_device()
    frame.to_device(device)
    frame_enc.to_device(device)
    # Deactivate wandb
    os.environ["WANDB_MODE"] = "disabled"
    wandb.init()
    assert training_phase.end_lr is not None  # To make pyright happy.
    frame_enc = coolchic_train(
        frame_encoder=frame_enc,
        frame=frame,
        frame_encoder_manager=frame_encoder_manager,
        start_lr=training_phase.lr,
        end_lr=training_phase.end_lr,
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


# 5: Get metrics at different points of training (post-dec rate as well)
if __name__ == "__main__":
    # Configuring how training happens.
    training_phase = TrainerPhase(
        lr=1e-3,
        end_lr=1e-5,
        schedule_lr=True,
        max_itr=5000,
        freq_valid=10,
        patience=1000,
        optimized_module=["all"],
        quantizer_type="softround",
        quantizer_noise_type="gaussian",
        softround_temperature=(0.3, 0.1),
        noise_parameter=(0.25, 0.1),
        quantize_model=True,
    )
    training_preset = PresetConfig(
        preset_name="", warmup=Warmup(), all_phases=[training_phase]
    )
    main(1, training_preset)
    # main(1, training_preset, from_scratch=True)

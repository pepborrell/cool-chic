from coolchic.enc.component.coolchic import CoolChicEncoder, CoolChicEncoderParameter
from coolchic.enc.component.video import VideoEncoder
from coolchic.enc.utils.codingstructure import CodingStructure, FrameData
from coolchic.enc.utils.manager import FrameEncoderManager
from coolchic.enc.utils.parsecli import get_coolchic_param_from_args
from coolchic.encode_simpler import build_frame_encoder
from coolchic.utils.types import DecoderConfig, PresetConfig


def get_coolchic_structs(
    lmbda: float,
    preset_config: PresetConfig,
    dec_cfg: DecoderConfig,
    cc_encoder: CoolChicEncoder,
    frame_data: FrameData,
):
    # Fully fledged coolchic representation to go to training
    coding_structure = CodingStructure(intra_period=0)
    frame_encoder_manager = FrameEncoderManager(
        preset_config=preset_config, lmbda=lmbda
    )
    coolchic_encoder_parameter = CoolChicEncoderParameter(
        **get_coolchic_param_from_args(dec_cfg)
    )
    frame = coding_structure.get_frame_from_coding_order(0)
    assert frame is not None  # To make pyright happy.
    frame.data = frame_data
    coolchic_encoder_parameter.set_image_size(frame.data.img_size)
    frame_enc = build_frame_encoder(coolchic_encoder_parameter, frame)
    frame_enc.coolchic_encoder = cc_encoder

    # We need the video encoder for this one operation.
    # TODO: can we do this without the video encoder and drop it completely?
    video_encoder = VideoEncoder(
        coding_structure=coding_structure,
        shared_coolchic_parameter=coolchic_encoder_parameter,
        shared_frame_encoder_manager=frame_encoder_manager,
    )
    frame.refs_data = video_encoder.get_ref_data(frame)

    return frame, frame_encoder_manager, frame_enc

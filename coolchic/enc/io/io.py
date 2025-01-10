import torch

from coolchic.enc.io.format.data_type import FRAME_DATA_TYPE, POSSIBLE_BITDEPTH
from coolchic.enc.io.format.png import read_png
from coolchic.enc.io.format.ppm import read_ppm
from coolchic.enc.io.format.yuv import read_yuv
from coolchic.enc.utils.codingstructure import FrameData


def load_frame_data_from_file(file_path: str, idx_display_order: int) -> FrameData:
    """Load the idx_display_order-th frame from a .yuv file or .png file. For the latter,
    idx_display_order must be equal to 0 as there is only one frame in a png.

    Args:
        file_path (str): Absolute path of the file from which the frame is loaded.
        idx_display_order (int): Index of the frame in display order

    Returns:
        FrameData: The loaded frame, wrapped as a FrameData object.
    """
    POSSIBLE_EXT = [".yuv", ".png", ".ppm"]
    assert file_path[-4:] in POSSIBLE_EXT, (
        "The function load_frame_data_from_file() expects a file ending with "
        f"{POSSIBLE_EXT}. Found {file_path}"
    )

    if file_path.endswith(".yuv"):
        # ! We only consider yuv420 and 444 planar
        bitdepth: POSSIBLE_BITDEPTH = 8 if "_8b" in file_path else 10
        frame_data_type: FRAME_DATA_TYPE = "yuv420" if "420" in file_path else "yuv444"
        data = read_yuv(file_path, idx_display_order, frame_data_type, bitdepth)

    elif file_path.endswith(".png"):
        frame_data_type: FRAME_DATA_TYPE = "rgb"
        data, bitdepth = read_png(file_path)

    elif file_path.endswith(".ppm"):
        frame_data_type: FRAME_DATA_TYPE = "rgb"
        data, bitdepth = read_ppm(file_path)

    return FrameData(bitdepth, frame_data_type, data)


def load_frame_data_from_tensor(data: torch.Tensor) -> FrameData:
    """Load a frame from a tensor."""
    frame_data_type = "rgb"
    bitdepth = 8
    return FrameData(bitdepth, frame_data_type, data)

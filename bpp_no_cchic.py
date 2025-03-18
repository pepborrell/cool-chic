import os
import subprocess
from pathlib import Path

import torch
from CCLIB.ccencapi import cc_code_latent_layer_bac, cc_code_wb_bac

from coolchic.dec.nn import decode_network
from coolchic.enc.bitstream.armint import ArmInt
from coolchic.enc.bitstream.encode import get_ac_max_val_latent, get_ac_max_val_nn
from coolchic.enc.bitstream.utils import get_sub_bitstream_path
from coolchic.enc.component.coolchic import CoolChicEncoder
from coolchic.enc.component.core.synthesis import Synthesis
from coolchic.enc.component.core.upsampling import Upsampling
from coolchic.enc.utils.codingstructure import FrameData
from coolchic.enc.utils.misc import (
    FIXED_POINT_FRACTIONAL_BITS,
    FIXED_POINT_FRACTIONAL_MULT,
    MAX_AC_MAX_VAL,
    POSSIBLE_Q_STEP,
    POSSIBLE_Q_STEP_SHIFT,
    DescriptorCoolChic,
    DescriptorNN,
)

_FRAME_DATA_TYPE = ["rgb", "yuv420", "yuv444"]
_POSSIBLE_BITDEPTH = [8, 9, 10, 11, 12, 13, 14, 15, 16]
_POSSIBLE_SYNTHESIS_MODE = [k for k in Synthesis.possible_mode]
_POSSIBLE_SYNTHESIS_NON_LINEARITY = [k for k in Synthesis.possible_non_linearity]


def n_bytes_in_header() -> int:
    """Header has fixed structure, and therefore a fixed number of bytes.
    This function returns the number of bytes in the header, while
    showing how this number is calculated.
    """
    n_bytes_header = 0
    n_bytes_header += 2  # Number of bytes header
    n_bytes_header += 2  # Image height
    n_bytes_header += 2  # Image width
    n_bytes_header += 1  # GOP type, frame data type, bitdepth
    n_bytes_header += 1  # intra period
    n_bytes_header += 1  # p period
    return n_bytes_header


def write_gop_header(cc_encoder: CoolChicEncoder, header_path: str):
    """Write a frame header to a a file located at <header_path>.
    The structure of the header is described above.

    Args:
        cc_encoder (CoolChicEncoder): CoolChicEncoder object
        img (torch.Tensor): Image tensor, from the image being encoded.
        header_path (str): Path to the header file
    """

    n_bytes_header = n_bytes_in_header()

    byte_to_write = b""
    byte_to_write += n_bytes_header.to_bytes(2, byteorder="big", signed=False)

    img_size = cc_encoder.param.img_size
    assert img_size is not None, "img_size is None, expected the value to be set"
    byte_to_write += img_size[0].to_bytes(2, byteorder="big", signed=False)
    byte_to_write += img_size[1].to_bytes(2, byteorder="big", signed=False)

    # Mock frame data to extract the data we need here.
    # The values we give it are the ones hardcoded in load_frame_data_from_tensor.
    frame_data = FrameData(bitdepth=8, frame_data_type="rgb", data=torch.empty(1))
    byte_to_write += (
        # Last 4 bits are for the bitdepth
        _POSSIBLE_BITDEPTH.index(frame_data.bitdepth) * 2**4
        # The first 4 bits are for the frame data type
        + _FRAME_DATA_TYPE.index(frame_data.frame_data_type)
    ).to_bytes(1, byteorder="big", signed=False)

    intra_period = 1  # Only 1 image, I-frame.
    p_period = 1  # Because we have only one image, we don't have any P-frame.
    byte_to_write += intra_period.to_bytes(1, byteorder="big", signed=False)
    byte_to_write += p_period.to_bytes(1, byteorder="big", signed=False)

    with open(header_path, "wb") as fout:
        fout.write(byte_to_write)

    if n_bytes_header != os.path.getsize(header_path):
        print("Invalid number of bytes in header!")
        print("expected", n_bytes_header)
        print("got", os.path.getsize(header_path))
        exit(1)


def write_frame_header(
    cc_enc: CoolChicEncoder,
    header_path: str,
    n_bytes_per_latent: list[int],
    q_step_index_nn: DescriptorCoolChic,
    scale_index_nn: DescriptorCoolChic,
    n_bytes_nn: DescriptorCoolChic,
    ac_max_val_nn: int,
    ac_max_val_latent: int,
    hls_sig_blksize: int,
):
    """Write a frame header to a a file located at <header_path>.
    The structure of the header is described above.

    Args:
        cc_enc (CoolChicEncoder): CoolChicEncoder object
        header_path (str): Path of the file where the header is written.
        n_bytes_per_latent (List[int]): Indicates the number of bytes for each 2D
            latent grid.
        q_step_index_nn (DescriptorCoolChic): Dictionary containing the index of the
            quantization step for the weight and bias of each network.
        scale_index_nn (DescriptorCoolChic): Dictionary containing the index of the
            scale parameter used during the entropy coding of the weight and bias
            of each network.
        n_bytes_nn (DescriptorCoolChic): Dictionary containing the number of bytes
            used for the weights and biases of each network
        ac_max_val_nn (int): The range coder AC_MAX_VAL parameters for entropy coding the NNs
        ac_max_val_latent (int): The range coder AC_MAX_VAL parameters for entropy coding the latents
    """

    n_bytes_header = 0
    n_bytes_header += 2  # Number of bytes header
    n_bytes_header += 1  # Display index of the frame

    n_bytes_header += (
        1  # Context size and Hidden layer dimension ARM, n. hidden layers ARM
    )

    n_bytes_header += 1  # (n_ups_kernel << 4)|(ups_k_size)
    n_bytes_header += 1  # (n_ups_preconcat_kernel << 4)|(ups_preconcat_k_size)

    n_bytes_header += 1  # Number of synthesis branches
    n_bytes_header += 1  # Number hidden layer Synthesis per branch
    # Hidden Synthesis layer out#, kernelsz, mode+nonlinearity
    n_bytes_header += 3 * len(cc_enc.layers_synthesis)
    n_bytes_header += 1  # Flow gain

    n_bytes_header += 2  # AC_MAX_VAL for neural networks
    n_bytes_header += 2  # AC_MAX_VAL for the latent variables
    n_bytes_header += 1  # hls_sig_blksize

    n_bytes_header += 1  # Index of the quantization step weight ARM
    n_bytes_header += 1  # Index of the quantization step bias ARM
    n_bytes_header += 1  # Index of the quantization step weight Upsampling
    n_bytes_header += 1  # Index of the quantization step bias Upsampling
    n_bytes_header += 1  # Index of the quantization step weight Synthesis
    n_bytes_header += 1  # Index of the quantization step bias Synthesis

    n_bytes_header += 1  # Index of scale entropy coding weight ARM
    n_bytes_header += 1  # Index of scale entropy coding bias ARM
    n_bytes_header += 1  # Index of scale entropy coding weight Upsampling
    n_bytes_header += 1  # Index of scale entropy coding bias Upsampling
    n_bytes_header += 1  # Index of scale entropy coding weight Synthesis
    n_bytes_header += 1  # Index of scale entropy coding bias Synthesis

    n_bytes_header += 2  # Number of bytes for weight ARM
    n_bytes_header += 2  # Number of bytes for bias ARM
    n_bytes_header += 2  # Number of bytes for weight Upsampling
    n_bytes_header += 2  # Index of scale entropy coding bias Upsampling
    n_bytes_header += 2  # Number of bytes for weight Synthesis
    n_bytes_header += 2  # Number of bytes for bias Synthesis

    n_bytes_header += 1  # Number of latent resolutions
    n_bytes_header += 1  # Number of 2D latent grids
    n_bytes_header += 1 * len(
        cc_enc.latent_grids
    )  # Number of feature maps for each latent resolutions
    n_bytes_header += 3 * len(
        n_bytes_per_latent
    )  # Number of bytes for each 2D latent grid

    byte_to_write = b""
    byte_to_write += n_bytes_header.to_bytes(2, byteorder="big", signed=False)
    frame_display_order = 0  # Only one frame, so display order is 0.
    byte_to_write += frame_display_order.to_bytes(1, byteorder="big", signed=False)

    # Since the dimension of the hidden layer and of the context is always a multiple of
    # 8, we can spare 3 bits by dividing it by 8
    assert cc_enc.param.dim_arm // 8 < 2**4, (
        f"Number of context pixels"
        f" and dimension of the hidden layer for the arm must be inferior to {2 ** 4 * 8}. Found"
        f" {cc_enc.param.dim_arm}"
    )

    assert cc_enc.param.n_hidden_layers_arm < 2**4, (
        f"Number of hidden layers"
        f" for the ARM should be inferior to {2 ** 4}. Found "
        f"{cc_enc.param.n_hidden_layers_arm}"
    )

    byte_to_write += (
        (cc_enc.param.dim_arm // 8) * 2**4 + cc_enc.param.n_hidden_layers_arm
    ).to_bytes(1, byteorder="big", signed=False)

    byte_to_write += (
        # (frame_encoder.coolchic_encoder_param.n_ups_kernel<<4)|(frame_encoder.coolchic_encoder_param.ups_k_size)
        ((cc_enc.param.latent_n_grids - 1) << 4) | (cc_enc.param.ups_k_size)
    ).to_bytes(1, byteorder="big", signed=False)
    byte_to_write += (
        # (frame_encoder.coolchic_encoder_param.n_ups_preconcat_kernel<<4)|(frame_encoder.coolchic_encoder_param.ups_preconcat_k_size)
        ((cc_enc.param.latent_n_grids - 1) << 4) | (cc_enc.param.ups_preconcat_k_size)
    ).to_bytes(1, byteorder="big", signed=False)

    # Continue to send this byte for compatibility
    _dummy_n_synth_branch = 1
    byte_to_write += (
        _dummy_n_synth_branch  # frame_encoder.coolchic_encoder_param.n_synth_branch
    ).to_bytes(1, byteorder="big", signed=False)
    byte_to_write += len(cc_enc.param.layers_synthesis).to_bytes(
        1, byteorder="big", signed=False
    )
    # If no hidden layers in the Synthesis, frame_encoder.coolchic_encoder_param.layers_synthesis is an empty list. So write 0
    for layer_spec in cc_enc.param.layers_synthesis:
        out_ft, k_size, mode, non_linearity = layer_spec.split("-")
        byte_to_write += int(out_ft).to_bytes(1, byteorder="big", signed=False)
        byte_to_write += int(k_size).to_bytes(1, byteorder="big", signed=False)
        byte_to_write += (
            _POSSIBLE_SYNTHESIS_MODE.index(mode) * 16
            + _POSSIBLE_SYNTHESIS_NON_LINEARITY.index(non_linearity)
        ).to_bytes(1, byteorder="big", signed=False)

    flow_gain = 1.0  # This is how it is set in the intercoding.py file, I don't know if it is relevant.

    if isinstance(flow_gain, float):
        assert flow_gain.is_integer(), f"Flow gain should be integer, found {flow_gain}"
        flow_gain = int(flow_gain)

    assert (
        flow_gain >= 0 and flow_gain <= 255
    ), f"Flow gain should be in [0, 255], found {flow_gain}"
    byte_to_write += flow_gain.to_bytes(1, byteorder="big", signed=False)

    if ac_max_val_nn > MAX_AC_MAX_VAL:
        print("AC_MAX_VAL NN is too big!")
        print(f"Found {ac_max_val_nn}, should be smaller than {MAX_AC_MAX_VAL}")
        print("Exiting!")
        return
    if ac_max_val_latent > MAX_AC_MAX_VAL:
        print("AC_MAX_VAL latent is too big!")
        print(f"Found {ac_max_val_latent}, should be smaller than {MAX_AC_MAX_VAL}")
        print("Exiting!")
        return

    byte_to_write += ac_max_val_nn.to_bytes(2, byteorder="big", signed=False)
    byte_to_write += ac_max_val_latent.to_bytes(2, byteorder="big", signed=False)
    byte_to_write += hls_sig_blksize.to_bytes(1, byteorder="big", signed=True)

    for nn_name in ["arm", "upsampling", "synthesis"]:
        for nn_param in ["weight", "bias"]:
            cur_q_step_index = q_step_index_nn.get(nn_name).get(nn_param)
            byte_to_write += cur_q_step_index.to_bytes(1, byteorder="big", signed=False)

    for nn_name in ["arm", "upsampling", "synthesis"]:
        for nn_param in ["weight", "bias"]:
            cur_q_step_index = q_step_index_nn.get(nn_name).get(nn_param)
            cur_scale_index = scale_index_nn.get(nn_name).get(nn_param)
            byte_to_write += cur_scale_index.to_bytes(1, byteorder="big", signed=False)

    for nn_name in ["arm", "upsampling", "synthesis"]:
        for nn_param in ["weight", "bias"]:
            cur_q_step_index = q_step_index_nn.get(nn_name).get(nn_param)
            cur_n_bytes = n_bytes_nn.get(nn_name).get(nn_param)
            if cur_n_bytes > MAX_AC_MAX_VAL:
                print(f"Number of bytes for {nn_name} {nn_param} is too big!")
                print(f"Found {cur_n_bytes}, should be smaller than {MAX_AC_MAX_VAL}")
                print("Exiting!")
                return
            byte_to_write += cur_n_bytes.to_bytes(2, byteorder="big", signed=False)

    byte_to_write += cc_enc.param.latent_n_grids.to_bytes(
        1, byteorder="big", signed=False
    )
    byte_to_write += len(n_bytes_per_latent).to_bytes(1, byteorder="big", signed=False)

    for i, latent_i in enumerate(cc_enc.latent_grids):
        n_ft_i = latent_i.size()[1]
        if n_ft_i > 2**8 - 1:
            print(f"Number of feature maps for latent {i} is too big!")
            print(f"Found {n_ft_i}, should be smaller than {2 ** 8 - 1}")
            print("Exiting!")
            return
        byte_to_write += n_ft_i.to_bytes(1, byteorder="big", signed=False)

    for i, v in enumerate(n_bytes_per_latent):
        if v > 2**24 - 1:
            print(f"Number of bytes for latent {i} is too big!")
            print(f"Found {v}, should be smaller than {2 ** 24 - 1}")
            print("Exiting!")
            return
        # for tmp in n_bytes_per_latent:
        byte_to_write += v.to_bytes(3, byteorder="big", signed=False)

    with open(header_path, "wb") as fout:
        fout.write(byte_to_write)

    if n_bytes_header != os.path.getsize(header_path):
        print("Invalid number of bytes in header!")
        print("expected", n_bytes_header)
        print("got", os.path.getsize(header_path))
        exit(1)


@torch.no_grad()
def encode_frame(
    cc_enc: CoolChicEncoder,
    bitstream_path: str,
    idx_coding_order: int,
    hls_sig_blksize: int,
):
    """Convert a model to a bitstream located at <bitstream_path>.

    Args:
        model (CoolChicEncoder): A trained and quantized model
        bitstream_path (str): Where to save the bitstream
    """

    torch.use_deterministic_algorithms(True)
    cc_enc.eval()
    cc_enc.to_device("cpu")

    # upsampling has bias parameters, but we do not use them.
    have_bias = {
        "arm": True,
        "upsampling": False,
        "synthesis": True,
    }

    subprocess.call(f"rm -f {bitstream_path}", shell=True)

    # Move to pure-int Arm.  Transfer the quantized weights from the fp Arm.
    arm_fp_param = cc_enc.arm.get_param()
    arm_int = ArmInt(
        cc_enc.param.dim_arm,
        cc_enc.param.n_hidden_layers_arm,
        FIXED_POINT_FRACTIONAL_MULT,
        pure_int=True,
    )
    cc_enc.arm = arm_int  # pyright: ignore
    cc_enc.arm.set_param_from_float(arm_fp_param)

    # ================= Encode the MLP into a bitstream file ================ #
    ac_max_val_nn = get_ac_max_val_nn(cc_enc)

    # scale_index_nn: DescriptorCoolChic = {}
    # q_step_index_nn: DescriptorCoolChic = {}
    # n_bytes_nn: DescriptorCoolChic = {}
    scale_index_nn = {}
    q_step_index_nn = {}
    n_bytes_nn = {}
    for cur_module_name in cc_enc.modules_to_send:
        # Prepare to store values dedicated to the current modules
        scale_index_nn[cur_module_name] = {}
        q_step_index_nn[cur_module_name] = {}
        n_bytes_nn[cur_module_name] = {}

        module_to_encode = getattr(cc_enc, cur_module_name)

        weights, bias = [], []
        # Retrieve all the weights and biases for the ARM MLP
        q_step_index_nn[cur_module_name]["weight"] = -1
        q_step_index_nn[cur_module_name]["bias"] = -1
        for k, v in module_to_encode.named_parameters():
            assert cur_module_name in ["arm", "synthesis", "upsampling"], (
                f"Unknow module name {cur_module_name}. "
                'Module name should be in ["arm", "synthesis", "upsampling"].'
            )

            Q_STEPS = POSSIBLE_Q_STEP.get(cur_module_name)

            if "weight" in k:
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_possible_q_step = POSSIBLE_Q_STEP.get(cur_module_name).get("weight")
                cur_q_step = cc_enc.nn_q_step.get(cur_module_name).get("weight")
                cur_q_step_index = int(
                    torch.argmin((cur_possible_q_step - cur_q_step).abs()).item()
                )

                # Store it into q_step_index_nn. It is overwritten for each
                # loop but it does not matter
                q_step_index_nn[cur_module_name]["weight"] = cur_q_step_index

                # Quantize the weight with the actual quantization step and add it
                # to the list of (quantized) weights
                if cur_module_name == "arm":
                    # Our weights are stored as fixed point, we use shifts to get the integer values of quantized results.
                    # Our int vals are int(floatval << FPFBITS)
                    q_step_shift = abs(
                        POSSIBLE_Q_STEP_SHIFT["arm"]["weight"][cur_q_step_index]
                    )
                    delta = int(FIXED_POINT_FRACTIONAL_BITS - q_step_shift)
                    if delta > 0:
                        pos_v = (
                            v >> delta
                        )  # a following <<delta would be the actual weight.
                        neg_v = -(
                            -v >> delta
                        )  # a following <<delta would be the actual weight.
                        v = torch.where(v < 0, neg_v, pos_v)
                    weights.append(v.flatten())
                else:
                    # No longer relevant without the bi-branch synth
                    # # # Blending -- we get the transformed weight, not the underlying sigmoid parameter.
                    # # # plus: only if >1 branch.
                    # # if cur_module_name == "synthesis" and k.endswith(".parametrizations.weight.original"):
                    # #     if "branch_blender" in k and frame_encoder.coolchic_encoder_param.n_synth_branch == 1:
                    # #         continue # Do not emit unused blender weight.
                    # #     xformed_weights = getattr(module_to_encode, k.replace(".parametrizations.weight.original", "")).weight
                    # #     v = xformed_weights
                    weights.append(
                        torch.round(v / cur_possible_q_step[cur_q_step_index]).flatten()
                    )

            elif "bias" in k and have_bias[cur_module_name]:
                # Find the index of the closest quantization step in the list of
                # the Q_STEPS quantization step.
                cur_possible_q_step = POSSIBLE_Q_STEP.get(cur_module_name).get("bias")
                cur_q_step = cc_enc.nn_q_step.get(cur_module_name).get("bias")
                cur_q_step_index = int(
                    torch.argmin((cur_possible_q_step - cur_q_step).abs()).item()
                )

                # Store it into q_step_index_nn. It is overwritten for each
                # loop but it does not matter
                q_step_index_nn[cur_module_name]["bias"] = cur_q_step_index

                # Quantize the bias with the actual quantization step and add it
                # to the list of (quantized) bias
                if cur_module_name == "arm":
                    # Our biases are stored as fixed point, we use shifts to get the integer values of quantized results.
                    # Our int vals are int(floatval << FPFBITS << FPFBITS)
                    q_step_shift = abs(
                        POSSIBLE_Q_STEP_SHIFT["arm"]["bias"][cur_q_step_index]
                    )
                    delta = int(FIXED_POINT_FRACTIONAL_BITS * 2 - q_step_shift)
                    if delta > 0:
                        pos_v = (
                            v >> delta
                        )  # a following <<delta would be the actual weight.
                        neg_v = -(
                            -v >> delta
                        )  # a following <<delta would be the actual weight.
                        v = torch.where(v < 0, neg_v, pos_v)
                    bias.append(v.flatten())
                else:
                    bias.append(
                        torch.round(v / cur_possible_q_step[cur_q_step_index]).flatten()
                    )

        # Gather them
        weights = torch.cat(weights).flatten()
        if have_bias[cur_module_name]:
            bias = torch.cat(bias).flatten()
        else:
            q_step_index_nn[cur_module_name]["bias"] = (
                0  # we actually send this in the header.
            )

        # ----------------- Actual entropy coding
        # It happens on cpu
        weights = weights.cpu()
        if have_bias[cur_module_name]:
            bias = bias.cpu()

        cur_bitstream_path = f"{bitstream_path}_{cur_module_name}_weight"

        # either code directly (normal), or search for best (backwards compatible).
        scale_index_weight = cc_enc.nn_expgol_cnt[cur_module_name]["weight"]
        if scale_index_weight is None:
            scale_index_weight = -1  # Search for best.
        scale_index_weight = cc_code_wb_bac(
            cur_bitstream_path,
            weights.flatten().to(torch.int32).tolist(),
            scale_index_weight,  # search for best count if -1
        )
        scale_index_nn[cur_module_name]["weight"] = scale_index_weight

        n_bytes_nn[cur_module_name]["weight"] = os.path.getsize(cur_bitstream_path)

        if have_bias[cur_module_name]:
            cur_bitstream_path = f"{bitstream_path}_{cur_module_name}_bias"

            # either code directly (normal), or search for best (backwards compatible).
            scale_index_bias = cc_enc.nn_expgol_cnt[cur_module_name]["bias"]
            if scale_index_bias is None:
                scale_index_bias = -1  # Search for best.
            scale_index_bias = cc_code_wb_bac(
                cur_bitstream_path,
                bias.flatten().to(torch.int32).tolist(),
                scale_index_bias,  # search for best count if -1
            )
            scale_index_nn[cur_module_name]["bias"] = scale_index_bias

            n_bytes_nn[cur_module_name]["bias"] = os.path.getsize(cur_bitstream_path)
        else:
            scale_index_nn[cur_module_name]["bias"] = 0
            n_bytes_nn[cur_module_name]["bias"] = 0
    # ================= Encode the MLP into a bitstream file ================ #

    # =============== Encode the latent into a bitstream file =============== #
    # To ensure perfect reproducibility between the encoder and the decoder,
    # we load the the different sub-networks from the bitstream here.
    for module_name in cc_enc.modules_to_send:
        assert module_name in ["arm", "synthesis", "upsampling"], (
            f"Unknow module name {module_name}. "
            'Module name should be in ["arm", "synthesis", "upsampling"].'
        )

        if module_name == "arm":
            empty_module = ArmInt(
                cc_enc.param.dim_arm,
                cc_enc.param.n_hidden_layers_arm,
                FIXED_POINT_FRACTIONAL_MULT,
                pure_int=True,
            )
        elif module_name == "synthesis":
            empty_module = Synthesis(
                sum(cc_enc.param.n_ft_per_res),
                cc_enc.param.layers_synthesis,
            )
        elif module_name == "upsampling":
            empty_module = Upsampling(
                cc_enc.param.ups_k_size,
                cc_enc.param.ups_preconcat_k_size,
                # frame_encoder.coolchic_encoder.param.n_ups_kernel,
                cc_enc.param.latent_n_grids - 1,
                # frame_encoder.coolchic_encoder.param.n_ups_preconcat_kernel,
                cc_enc.param.latent_n_grids - 1,
            )
        else:
            raise ValueError(f"Unknown module name {module_name}")

        Q_STEPS = POSSIBLE_Q_STEP.get(module_name)

        loaded_module = decode_network(
            empty_module,
            DescriptorNN(
                weight=f"{bitstream_path}_{module_name}_weight",
                bias=f"{bitstream_path}_{module_name}_bias"
                if have_bias[module_name]
                else "",
            ),
            DescriptorNN(
                weight=Q_STEPS["weight"][q_step_index_nn[module_name]["weight"]],
                bias=Q_STEPS["bias"][q_step_index_nn[module_name]["bias"]],
            ),
            DescriptorNN(
                scale_index_nn[module_name]["weight"],
                bias=(scale_index_nn[module_name]["bias"])
                if have_bias[module_name]
                else 0,
            ),
            ac_max_val_nn,
        )
        setattr(cc_enc, module_name, loaded_module)

    cc_enc.to_device("cpu")
    cc_enc.eval()

    ac_max_val_latent = get_ac_max_val_latent(cc_enc)

    # Setting visu to true allows to recover 2D mu, scale and latents

    _, _, additional_data = cc_enc.forward(
        quantizer_noise_type="none",
        quantizer_type="hardround",
        AC_MAX_VAL=ac_max_val_latent,
        flag_additional_outputs=True,
    )

    # Encode the different 2d latent grids one after the other
    n_bytes_per_latent = []
    torch.set_printoptions(threshold=10000000)
    ctr_2d_ft = 0
    # Loop on the different resolutions
    for index_lat_resolution in range(cc_enc.param.latent_n_grids):
        current_mu = additional_data.get("detailed_mu")[index_lat_resolution]
        # NOTE: variable current_scale is not used. Commenting out in case it's needed later on.
        # current_scale = encoder_output.additional_data.get("detailed_scale")[
        #     index_lat_resolution
        # ]
        current_log_scale = additional_data.get("detailed_log_scale")[
            index_lat_resolution
        ]
        current_y = additional_data.get("detailed_sent_latent")[index_lat_resolution]

        c_i, h_i, w_i = current_y.size()[-3:]

        # Nothing to send!
        if c_i == 0:
            n_bytes_per_latent.append(0)
            cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
            # Still create an empty file for coherence
            subprocess.call(f"touch {cur_latent_bitstream}", shell=True)
            ctr_2d_ft += 1
            continue

        # Loop on the different 2D grids composing one resolutions
        for index_lat_feature in range(c_i):
            y_this_ft = current_y[:, index_lat_feature, :, :].flatten().cpu()
            mu_this_ft = current_mu[:, index_lat_feature, :, :].flatten().cpu()
            log_scale_this_ft = (
                current_log_scale[:, index_lat_feature, :, :].flatten().cpu()
            )

            if y_this_ft.abs().max() == 0:
                n_bytes_per_latent.append(0)
                cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
                # Still create an empty file for coherence
                subprocess.call(f"touch {cur_latent_bitstream}", shell=True)
                ctr_2d_ft += 1
                continue

            cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
            cc_code_latent_layer_bac(
                cur_latent_bitstream,
                y_this_ft.flatten().to(torch.int32).tolist(),
                (mu_this_ft * FIXED_POINT_FRACTIONAL_MULT)
                .round()
                .flatten()
                .to(torch.int32)
                .tolist(),
                (log_scale_this_ft * FIXED_POINT_FRACTIONAL_MULT)
                .round()
                .flatten()
                .to(torch.int32)
                .tolist(),
                h_i,
                w_i,
                hls_sig_blksize,
            )
            n_bytes_per_latent.append(os.path.getsize(cur_latent_bitstream))

            ctr_2d_ft += 1

    # Write the header
    header_path = f"{bitstream_path}_header"
    write_frame_header(
        cc_enc,
        header_path,
        n_bytes_per_latent,
        q_step_index_nn,
        scale_index_nn,
        n_bytes_nn,
        ac_max_val_nn,
        ac_max_val_latent,
        hls_sig_blksize,
    )

    # Concatenate everything inside a single file
    subprocess.call(f"rm -f {bitstream_path}", shell=True)
    subprocess.call(f"cat {header_path} >> {bitstream_path}", shell=True)
    subprocess.call(f"rm -f {header_path}", shell=True)

    for cur_module_name in ["arm", "upsampling", "synthesis"]:
        for parameter_type in ["weight", "bias"]:
            cur_bitstream = f"{bitstream_path}_{cur_module_name}_{parameter_type}"
            if os.path.exists(cur_bitstream):
                subprocess.call(f"cat {cur_bitstream} >> {bitstream_path}", shell=True)
                subprocess.call(f"rm -f {cur_bitstream}", shell=True)

    ctr_2d_ft = 0
    for index_lat_resolution in range(cc_enc.param.latent_n_grids):
        # No feature: still increment the counter and remove the temporary bitstream file
        if cc_enc.latent_grids[index_lat_resolution].size()[1] == 0:
            cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
            subprocess.call(f"rm -f {cur_latent_bitstream}", shell=True)
            ctr_2d_ft += 1

        for index_lat_feature in range(
            cc_enc.latent_grids[index_lat_resolution].size()[1]
        ):
            cur_latent_bitstream = get_sub_bitstream_path(bitstream_path, ctr_2d_ft)
            subprocess.call(
                f"cat {cur_latent_bitstream} >> {bitstream_path}", shell=True
            )
            subprocess.call(f"rm -f {cur_latent_bitstream}", shell=True)
            ctr_2d_ft += 1

    # Encoding's done, we no longer need deterministic algorithms
    torch.use_deterministic_algorithms(False)


def encode_coolchic(cc_enc: CoolChicEncoder, bitstream_path: Path):
    # ======================== GOP HEADER ======================== #
    # Write the header
    # header_path = f"{bitstream_path}_gop_header"
    header_path = bitstream_path.with_stem(
        f"{bitstream_path.stem}_gop_header"
    ).with_suffix("")
    write_gop_header(cc_enc, str(header_path))

    # Concatenate everything inside a single file
    # subprocess.call(f"rm -f {bitstream_path}", shell=True)
    bitstream_path.unlink(missing_ok=True)
    subprocess.call(f"cat {header_path} >> {bitstream_path}", shell=True)
    # subprocess.call(f"rm -f {header_path}", shell=True)
    header_path.unlink(missing_ok=True)
    # ======================== GOP HEADER ======================== #

    idx_coding_order = 0  # Only encoding one image.

    frame_bitstream_path = str(
        bitstream_path.with_stem(f"{bitstream_path.stem}_idx_coding_order").with_suffix(
            ""
        )
    )
    encode_frame(
        cc_enc,
        frame_bitstream_path,
        idx_coding_order,
        hls_sig_blksize=16,  # don't know why, but it's always 16.
    )
    subprocess.call(f"cat {frame_bitstream_path} >> {bitstream_path}", shell=True)
    subprocess.call(f"rm -f {frame_bitstream_path}", shell=True)
    ###### endif

    real_rate_byte = os.path.getsize(bitstream_path)
    # Not very elegant but look at the first frame cool-chic to get the video resolution
    assert (
        cc_enc.param.img_size is not None
    ), "Image size not declared in encoder parameters."
    h, w = cc_enc.param.img_size
    n_frames = 1
    real_rate_bpp = real_rate_byte * 8 / (h * w * n_frames)
    print(f"Real rate        [kBytes]: {real_rate_byte / 1000:9.3f}")
    print(f"Real rate           [bpp]: {real_rate_bpp :9.3f}")

    return real_rate_bpp

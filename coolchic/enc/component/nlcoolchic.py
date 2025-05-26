import typing
from dataclasses import fields
from typing import Any, OrderedDict

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch import nn

from coolchic.enc.component.coolchic import CoolChicEncoder, CoolChicEncoderParameter
from coolchic.enc.component.core.arm import (
    Arm,
    _get_neighbor,
    _get_non_zero_pixel_ctx_index,
    _laplace_cdf,
)
from coolchic.enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
    quantize,
)
from coolchic.enc.component.core.synthesis import Synthesis
from coolchic.enc.component.core.upsampling import Upsampling
from coolchic.enc.utils.misc import (
    MAX_ARM_MASK_SIZE,
    POSSIBLE_DEVICE,
    DescriptorCoolChic,
    DescriptorNN,
    measure_expgolomb_rate,
)
from coolchic.enc.visu.console import pretty_string_nn, pretty_string_ups


class LatentFreeCoolChicEncoder(nn.Module):
    """CoolChicEncoder for a single frame."""

    def __init__(self, param: CoolChicEncoderParameter):
        """Instantiate a cool-chic encoder for one frame.

        Args:
            param (CoolChicEncoderParameter): Architecture of the
                `CoolChicEncoder`. See the documentation of
                `CoolChicEncoderParameter` for more information
        """
        super().__init__()

        # Everything is stored inside param
        self.param = param

        assert self.param.img_size is not None, (
            "You are trying to instantiate a CoolChicEncoder from a "
            "CoolChicEncoderParameter with a field img_size set to None. Use "
            "the function coolchic_encoder_param.set_img_size((H, W)) before "
            "instantiating the CoolChicEncoder."
        )

        # ================== Synthesis related stuff ================= #
        # Encoder-side latent gain applied prior to quantization, one per feature
        self.encoder_gains = param.encoder_gain

        # Instantiate the synthesis MLP with as many inputs as the number
        # of latent channels
        self.synthesis = Synthesis(param.latent_n_grids, self.param.layers_synthesis)
        # ================== Synthesis related stuff ================= #

        # ===================== Upsampling stuff ===================== #
        self.upsampling = Upsampling(
            ups_k_size=self.param.ups_k_size,
            ups_preconcat_k_size=self.param.ups_preconcat_k_size,
            # Instantiate one different upsampling and pre-concatenation
            # filters for each of the upsampling step. Could also be set to one
            # to share the same filter across all latents.
            n_ups_kernel=self.param.latent_n_grids - 1,
            n_ups_preconcat_kernel=self.param.latent_n_grids - 1,
        )
        # ===================== Upsampling stuff ===================== #

        # ===================== ARM related stuff ==================== #
        # Create the probability model for the main INR. It uses a spatial context
        # parameterized by the spatial context

        # For a given mask size N (odd number e.g. 3, 5, 7), we have at most
        # (N * N - 1) / 2 context pixels in it.
        # Example, a 9x9 mask as below has 40 context pixel (indicated with 1s)
        # available to predict the pixel '*'
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 * 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0 0 0

        # No more than 40 context pixels i.e. a 9x9 mask size (see example above)
        max_mask_size = MAX_ARM_MASK_SIZE
        max_context_pixel = int((max_mask_size**2 - 1) / 2)
        assert self.param.dim_arm <= max_context_pixel, (
            f"You can not have more context pixels "
            f" than {max_context_pixel}. Found {self.param.dim_arm}"
        )

        # Mask of size 2N + 1 when we have N rows & columns of context.
        self.mask_size = max_mask_size

        # 1D tensor containing the indices of the selected context pixels.
        # register_buffer for automatic device management. We set persistent to false
        # to simply use the "automatically move to device" function, without
        # considering non_zero_pixel_ctx_index as a parameters (i.e. returned
        # by self.parameters())
        self.register_buffer(
            "non_zero_pixel_ctx_index",
            _get_non_zero_pixel_ctx_index(self.param.dim_arm),
            persistent=False,
        )

        self.arm = Arm(self.param.dim_arm, self.param.n_hidden_layers_arm)
        # ===================== ARM related stuff ==================== #

        # Something like ['arm', 'synthesis', 'upsampling']
        self.modules_to_send = [tmp.name for tmp in fields(DescriptorCoolChic)]

        # ======================== Monitoring ======================== #
        # Pretty string representing the decoder complexity
        self.flops_str = ""
        # Total number of multiplications to decode the image
        self.total_flops = 0
        self.flops_per_module = {k: 0 for k in self.modules_to_send}
        # Fill the two attributes aboves
        self.get_flops()
        # ======================== Monitoring ======================== #

        # Track the quantization step of each neural network, None if the
        # module is not yet quantized
        self.nn_q_step: dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

        # Track the exponent of the exp-golomb code used for the NN parameters.
        # None if module is not yet quantized
        self.nn_expgol_cnt: dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

        # Copy of the full precision parameters, set just before calling the
        # quantize_model() function. This is done through the
        # self._store_full_precision_param() function
        self.full_precision_param = None

    # ------- Actual forward
    def forward(
        self,
        latents: list[torch.Tensor],
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: torch.Tensor | None = torch.tensor(0.3),
        noise_parameter: torch.Tensor | None = torch.tensor(1.0),
        AC_MAX_VAL: int = -1,
        flag_additional_outputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Perform CoolChicEncoder forward pass, to be used during the training.
        The main step are as follows:

            1. **Scale & quantize the encoder-side latent** :math:`\\mathbf{y}` to
               get the decoder-side latent

                .. math::

                    \\hat{\\mathbf{y}} = \\mathrm{Q}(\\Gamma_{enc}\\ \\mathbf{y}),

                with :math:`\\Gamma_{enc} \\in \\mathbb{R}` a scalar encoder gain
                defined in ``self.param.encoder_gains`` and :math:`\\mathrm{Q}`
                the :doc:`quantization operation <core/quantizer>`.

            2. **Measure the rate** of the decoder-side latent with the
               :doc:`ARM <core/arm>`:

                .. math::

                    \\mathrm{R}(\\hat{\\mathbf{y}}) = -\\log_2 p_{\\psi}(\\hat{\\mathbf{y}}),

               where :math:`p_{\\psi}`
               is given by the :doc:`Auto-Regressive Module (ARM) <core/arm>`.

            3. **Upsample and synthesize** the latent to get the output

                .. math::

                    \\hat{\\mathbf{x}} = f_{\\theta}(f_{\\upsilon}(\\hat{\\mathbf{y}})),

               with :math:`f_{\\psi}` the :doc:`Upsampling <core/upsampling>`
               and :math:`f_{\\theta}` the :doc:`Synthesis <core/synthesis>`.

        Args:
            quantizer_noise_type: Defaults to ``"kumaraswamy"``.
            quantizer_type: Defaults to ``"softround"``.
            soft_round_temperature: Soft round temperature.
                This is used for softround modes as well as the
                ste mode to simulate the derivative in the backward.
                Defaults to 0.3.
            noise_parameter: noise distribution parameter. Defaults to 1.0.
            AC_MAX_VAL: If different from -1, clamp the value to be in
                :math:`[-AC\\_MAX\\_VAL; AC\\_MAX\\_VAL + 1]` to write the actual bitstream.
                Defaults to -1.
            flag_additional_outputs: True to fill
                ``CoolChicEncoderOutput['additional_data']`` with many different
                quantities which can be used to analyze Cool-chic behavior.
                Defaults to False.

        Returns:
            Output of Cool-chic training forward pass.
        """

        # ! Order of the operations are important as these are asynchronous
        # ! CUDA operations. Some ordering are faster than other...

        # ------ Encoder-side: quantize the latent
        # Convert the N [B, C, H_i, W_i] 4d latents with different resolutions
        # to a 2d [B, N*C*H*W] tensor. This allows to call the quantization
        # only once, which is faster.
        B = latents[0].shape[0]

        encoder_side_flat_latent = torch.cat(
            [latent_i.flatten(start_dim=1) for latent_i in latents], dim=1
        )

        flat_decoder_side_latent = quantize(
            encoder_side_flat_latent * self.encoder_gains,
            quantizer_noise_type if self.training else "none",
            quantizer_type if self.training else "hardround",
            soft_round_temperature,
            noise_parameter,
        )

        # Clamp latent if we need to write a bitstream
        if AC_MAX_VAL != -1:
            flat_decoder_side_latent = torch.clamp(
                flat_decoder_side_latent, -AC_MAX_VAL, AC_MAX_VAL + 1
            )

        # Convert back the 1d tensor to a list of N [B, C, H_i, W_i] 4d latents.
        # This require a few additional information about each individual
        # latent dimension, stored in self.size_per_latent
        decoder_side_latent = []
        cnt = 0
        for latent in latents:
            _, c, h, w = latent.shape  # b should be one. or not if we batch
            latent_numel = c * h * w
            decoder_side_latent.append(
                flat_decoder_side_latent[:, cnt : cnt + latent_numel].reshape(
                    latent.shape
                )
            )
            cnt += latent_numel

        # ----- ARM to estimate the distribution and the rate of each latent
        # As for the quantization, we flatten all the latent and their context
        # so that the ARM network is only called once.
        # flat_latent: [B, N, 1] tensor describing N latents
        # flat_context: [B, N, context_size] tensor describing each latent context

        # Get all the context as a single 3D vector of size [B, M := N*H*W, context size]
        flat_context = torch.cat(
            [
                _get_neighbor(
                    spatial_latent_i, self.mask_size, self.non_zero_pixel_ctx_index
                )
                for spatial_latent_i in decoder_side_latent
            ],
            dim=1,
        )

        # Feed the spatial context to the arm MLP and get mu and scale
        flat_mu, flat_scale, flat_log_scale = self.arm.forward(flat_context)

        # Get all the M latent variables flat in one vector [B, M]
        flat_latent = torch.cat(
            [spatial_latent_i.view(B, -1) for spatial_latent_i in decoder_side_latent],
            dim=1,
        )

        # Compute the rate (i.e. the entropy of flat latent knowing mu and scale)
        proba = torch.clamp_min(
            _laplace_cdf(flat_latent + 0.5, flat_mu, flat_scale)
            - _laplace_cdf(flat_latent - 0.5, flat_mu, flat_scale),
            min=2**-16,  # No value can cost more than 16 bits.
        )
        flat_rate = -torch.log2(proba)  # shape: [B, M]

        # Upsampling and synthesis to get the output
        synthesis_output = self.synthesis(self.upsampling(decoder_side_latent))

        additional_data = {}
        if flag_additional_outputs:
            if B > 1:
                raise NotImplementedError(
                    "Batching is not yet supported for additional outputs."
                )

            # Prepare list to accommodate the visualisations
            additional_data["detailed_sent_latent"] = []
            additional_data["detailed_mu"] = []
            additional_data["detailed_scale"] = []
            additional_data["detailed_log_scale"] = []
            additional_data["detailed_rate_bit"] = []
            additional_data["detailed_centered_latent"] = []
            additional_data["hpfilters"] = []

            # "Pointer" for the reading of the 1D scale, mu and rate
            cnt = 0
            # for i, _ in enumerate(filtered_latent):
            for index_latent_res in range(len(latents)):
                c_i, h_i, w_i = decoder_side_latent[index_latent_res].size()[-3:]
                additional_data["detailed_sent_latent"].append(
                    decoder_side_latent[index_latent_res].view((1, c_i, h_i, w_i))
                )

                # Scale, mu and rate are 1D tensors where the N latent grids
                # are flattened together. As such we have to read the appropriate
                # number of values in this 1D vector to reconstruct the i-th grid in 2D
                mu_i, scale_i, log_scale_i, rate_i = [
                    # Read h_i * w_i values starting from cnt
                    tmp[cnt : cnt + (c_i * h_i * w_i)].view((1, c_i, h_i, w_i))
                    for tmp in [flat_mu, flat_scale, flat_log_scale, flat_rate]
                ]

                cnt += c_i * h_i * w_i
                additional_data["detailed_mu"].append(mu_i)
                additional_data["detailed_scale"].append(scale_i)
                additional_data["detailed_log_scale"].append(log_scale_i)
                additional_data["detailed_rate_bit"].append(rate_i)
                additional_data["detailed_centered_latent"].append(
                    additional_data["detailed_sent_latent"][-1] - mu_i
                )

        return synthesis_output, flat_rate, additional_data

    # ------- Getter / Setter and Initializer
    def get_param(self) -> OrderedDict[str, torch.Tensor]:
        """Return **a copy** of the weights and biases inside the module.

        Returns:
            OrderedDict[str, Tensor]: A copy of all weights & biases in the module.
        """
        param = OrderedDict({})
        param.update({f"arm.{k}": v for k, v in self.arm.get_param().items()})
        param.update(
            {f"upsampling.{k}": v for k, v in self.upsampling.get_param().items()}
        )
        param.update(
            {f"synthesis.{k}": v for k, v in self.synthesis.get_param().items()}
        )
        return param

    def set_param(self, param: OrderedDict[str, torch.Tensor]):
        """Replace the current parameters of the module with param.

        Args:
            param (OrderedDict[str, Tensor]): Parameters to be set.
        """
        self.load_state_dict(param)

    def initialize_latent_grids(
        self, zeros: bool = True, random_seed: int | None = None
    ) -> None:
        """This function is kept for compatibility, but this class doesn't have latent grids."""
        print(
            "WARNING: Latent grids should not be initialized when using a hypernetwork!"
        )

    def reinitialize_parameters(self):
        """Reinitialize in place the different parameters of a CoolChicEncoder
        namely the latent grids, the arm, the upsampling and the weights.
        """
        self.arm.reinitialize_parameters()
        self.upsampling.reinitialize_parameters()
        self.synthesis.reinitialize_parameters()

        # Reset the quantization steps and exp-golomb count of the neural
        # network to None since we are resetting the parameters.
        self.nn_q_step: dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }
        self.nn_expgol_cnt: dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

    def _store_full_precision_param(self) -> None:
        """Store the current parameters inside self.full_precision_param

        This function checks that there is no self.nn_q_step and
        self.nn_expgol_cnt already saved. This would mean that we no longer
        have full precision parameters but quantized ones.
        """

        if self.full_precision_param is not None:
            print(
                "Warning: overwriting already saved full-precision parameters"
                " in CoolChicEncoder _store_full_precision_param()."
            )

        # Check that we haven't already quantized the network by looking at
        # the nn_expgol_cnt and nn_q_step dictionaries
        no_q_step = True
        for _, q_step_dict in self.nn_q_step.items():
            for _, q_step in q_step_dict.items():
                if q_step is not None:
                    no_q_step = False
        assert no_q_step, (
            "Trying to store full precision parameters, while CoolChicEncoder "
            "nn_q_step attributes is not full of None. This means that the "
            "parameters have already been quantized... aborting!"
        )

        no_expgol_cnt = True
        for _, expgol_cnt_dict in self.nn_expgol_cnt.items():
            for _, expgol_cnt in expgol_cnt_dict.items():
                if expgol_cnt is not None:
                    no_expgol_cnt = False
        assert no_expgol_cnt, (
            "Trying to store full precision parameters, while CoolChicEncoder "
            "nn_expgol_cnt attributes is not full of None. This means that the "
            "parameters have already been quantized... aborting!"
        )

        # All good, simply save the parameters
        self.full_precision_param = self.get_param()

    def _load_full_precision_param(self) -> None:
        assert self.full_precision_param is not None, (
            "Trying to load full precision parameters but "
            "self.full_precision_param is None"
        )

        self.set_param(self.full_precision_param)

        # Reset the side information about the quantization step and expgol cnt
        # so that the rate is no longer computed by the test() function.
        self.nn_q_step: dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

        self.nn_expgol_cnt: dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

    # ------- Get flops, neural network rates and quantization step
    def get_flops(self) -> int:
        """Compute the number of MAC & parameters for the model.
        Update ``self.total_flops`` (integer describing the number of total MAC)
        and ``self.flops_str``, a pretty string allowing to print the model
        complexity somewhere.

        .. attention::

            ``fvcore`` measures MAC (multiplication & accumulation) but calls it
            FLOP (floating point operation)... We do the same here and call
            everything FLOP even though it would be more accurate to use MAC.

        Docstring taken from the original coolchic encoder implementation.
        """
        # print("Ignoring get_flops")
        # Count the number of floating point operations here. It must be done before
        # torch scripting the different modules.

        self = self.train(mode=False)

        assert (
            self.param.img_size is not None
        ), "Image size must be set in the parameter."
        mock_patch_size = self.param.img_size

        flops = FlopCountAnalysis(
            self,
            (
                [
                    torch.zeros(
                        (1, 1, mock_patch_size[0] // 2**i, mock_patch_size[1] // 2**i)
                    )
                    for i in range(7)
                ],  # latents
                "none",  # Quantization noise
                "hardround",  # Quantizer type
                0.3,  # Soft round temperature
                0.1,  # Noise parameter
                -1,  # AC_MAX_VAL
                False,  # Flag additional outputs
            ),
        )
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)

        self.total_flops = flops.total()
        for k in self.flops_per_module:
            self.flops_per_module[k] = flops.by_module()[k]

        self.flops_str = flop_count_table(flops)
        del flops

        self = self.train(mode=True)

        return self.total_flops

    def get_network_rate(self) -> DescriptorCoolChic:
        """Return the rate (in bits) associated to the parameters
        (weights and biases) of the different modules

        Returns:
            DescriptorCoolChic: The rate (in bits) associated with the weights
            and biases of each module
        """
        rate_per_module: DescriptorCoolChic = {
            module_name: {"weight": 0.0, "bias": 0.0}
            for module_name in self.modules_to_send
        }

        for module_name in self.modules_to_send:
            cur_module = getattr(self, module_name)
            rate_per_module[module_name] = measure_expgolomb_rate(
                cur_module,
                self.nn_q_step.get(module_name),
                self.nn_expgol_cnt.get(module_name),
            )
            if (
                module_name == "upsampling"
                and sum(rate_per_module[module_name].values()) > 0
            ):
                raise ValueError(
                    "Upsampling module should not have any quantized parameters."
                    " Please check the quantization step and exp-golomb count."
                )

        return rate_per_module

    def get_network_quantization_step(self) -> DescriptorCoolChic:
        """Return the quantization step associated to the parameters (weights
        and biases) of the different modules. Those quantization can be
        ``None`` if the model has not yet been quantized.

        Returns:
            DescriptorCoolChic: The quantization step associated with the
            weights and biases of each module.
        """
        return self.nn_q_step

    def get_network_expgol_count(self) -> DescriptorCoolChic:
        """Return the Exp-Golomb count parameter associated to the parameters
        (weights and biases) of the different modules. Those quantization can be
        ``None`` if the model has not yet been quantized.

        Returns:
            DescriptorCoolChic: The Exp-Golomb count parameter associated
            with the weights and biases of each module.
        """
        return self.nn_expgol_cnt

    def str_complexity(self) -> str:
        """Return a string describing the number of MAC (**not mac per pixel**) and the
        number of parameters for the different modules of CoolChic

        Returns:
            str: A pretty string about CoolChic complexity.
        """

        if not self.flops_str:
            self.get_flops()

        msg_total_mac = "----------------------------------\n"
        msg_total_mac += (
            f"Total MAC / decoded pixel: {self.get_total_mac_per_pixel():.1f}"
        )
        msg_total_mac += "\n----------------------------------"

        return self.flops_str + "\n\n" + msg_total_mac

    def get_total_mac_per_pixel(self) -> float:
        """Count the number of Multiplication-Accumulation (MAC) per decoded pixel
        for this model.

        Returns:
            float: number of floating point operations per decoded pixel.
        """

        if not self.flops_str:
            self.get_flops()

        n_pixels = self.param.img_size[-2] * self.param.img_size[-1]
        return self.total_flops / n_pixels

    # ------- Useful functions
    def to_device(self, device: POSSIBLE_DEVICE) -> None:
        """Push a model to a given device.

        Args:
            device (POSSIBLE_DEVICE): The device on which the model should run.
        """

        assert device in typing.get_args(
            POSSIBLE_DEVICE
        ), f"Unknown device {device}, should be in {typing.get_args(POSSIBLE_DEVICE)}"
        self = self.to(device)

        # Push integerized weights and biases of the mlp (resp qw and qb) to
        # the required device
        for idx_layer, layer in enumerate(self.arm.mlp):
            if hasattr(layer, "qw"):
                if layer.qw is not None:
                    self.arm.mlp[idx_layer].qw = layer.qw.to(device)

            if hasattr(layer, "qb"):
                if layer.qb is not None:
                    self.arm.mlp[idx_layer].qb = layer.qb.to(device)

    def pretty_string(self) -> str:
        """Get a pretty string representing the layer of a ``CoolChicEncoder``"""

        s = ""

        if not self.flops_str:
            self.get_flops()

        n_pixels = self.param.img_size[-2] * self.param.img_size[-1]
        total_mac_per_pix = self.get_total_mac_per_pixel()

        title = f"Cool-chic architecture {total_mac_per_pix:.0f} MAC / pixel"
        s += f"\n{title}\n" f"{'-' * len(title)}\n\n"

        complexity = self.flops_per_module["upsampling"] / n_pixels
        share_complexity = 100 * complexity / total_mac_per_pix
        title = f"Upsampling {complexity:.0f} MAC/pixel ; {share_complexity:.1f} % of the complexity"
        s += (
            f"{title}\n"
            f"{'=' * len(title)}\n"
            "Note: all upsampling layers are separable and symmetric "
            "(transposed) convolutions.\n\n"
        )
        s += pretty_string_ups(self.upsampling, "")

        complexity = self.flops_per_module["arm"] / n_pixels
        share_complexity = 100 * complexity / total_mac_per_pix
        title = f"ARM {complexity:.0f} MAC/pixel ; {share_complexity:.1f} % of the complexity"
        s += f"\n\n\n{title}\n" f"{'=' * len(title)}\n\n\n"
        input_arm = f"{self.arm.dim_arm}-pixel context"
        output_arm = "mu, log scale"
        s += pretty_string_nn(self.arm.mlp, "", input_arm, output_arm)

        complexity = self.flops_per_module["synthesis"] / n_pixels
        share_complexity = 100 * complexity / total_mac_per_pix
        title = f"Synthesis {complexity:.0f} MAC/pixel ; {share_complexity:.1f} % of the complexity"
        s += f"\n\n\n{title}\n" f"{'=' * len(title)}\n\n\n"
        input_syn = f"{self.synthesis.input_ft} features"
        output_syn = "Decoded image"
        s += pretty_string_nn(self.synthesis.layers, "", input_syn, output_syn)

        return s

    def as_coolchic(
        self,
        latents: list[torch.Tensor],
        stop_grads: bool = False,
        new_parameters: dict[str, torch.Tensor] | None = None,
    ) -> CoolChicEncoder:
        """Returns a CoolChicEncoder with the latents and deltas set."""
        if not stop_grads:
            raise NotImplementedError(
                "This method is only implemented for stop_grads=True."
            )

        encoder = CoolChicEncoder(self.param)

        # Replacing weights.
        # If we want to stop gradients, we need to make the tensors leaves in the graph.
        def state_dict_to_param(
            state: dict[str, torch.Tensor],
        ) -> OrderedDict[str, nn.Parameter]:
            # This should not happen if stop grads is True.
            return OrderedDict({k: nn.Parameter(v) for k, v in state.items()})

        if new_parameters is not None:

            def extract_component_params_from_state(
                params: dict[str, torch.Tensor], comp_prefix: str
            ) -> dict[str, torch.Tensor]:
                return {
                    k.removeprefix(comp_prefix): v
                    for k, v in params.items()
                    if k.startswith(comp_prefix)
                }

            synthesis_state_dict = extract_component_params_from_state(
                new_parameters, "synthesis."
            )
            arm_state_dict = extract_component_params_from_state(new_parameters, "arm.")
            upsampling_state_dict = extract_component_params_from_state(
                new_parameters, "upsampling."
            )
        else:
            synthesis_state_dict = self.synthesis.state_dict()
            arm_state_dict = self.arm.state_dict()
            upsampling_state_dict = self.upsampling.state_dict()

        encoder.synthesis.set_hypernet_weights(
            state_dict_to_param(synthesis_state_dict)
        )
        encoder.arm.set_hypernet_weights(state_dict_to_param(arm_state_dict))
        encoder.upsampling.set_hypernet_weights(
            state_dict_to_param(upsampling_state_dict)
        )
        # Replace latents in CoolChicEncoder.
        encoder.size_per_latent = [lat.shape for lat in latents]

        # Only because stop grads is True.
        latents = [nn.Parameter(lat) for lat in latents]
        # Just checking.
        assert all(
            lat.is_leaf for lat in latents
        ), "Latents are not leaves. They still carry gradients back."

        # Something like self.latent_grids = nn.ParameterList(latents)
        # would break the computation graph. This doesn't. Following tips in:
        # https://github.com/qu-gg/torch-hypernetwork-tutorials?tab=readme-ov-file#tensorVSparameter
        for i in range(len(latents)):
            del encoder.latent_grids[i].data
            encoder.latent_grids[i].data = latents[i]

        return encoder.to(next(self.parameters()).device)

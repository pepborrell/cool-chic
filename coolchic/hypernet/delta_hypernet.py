from typing import Any

import torch
from torch import nn

from coolchic.enc.component.coolchic import CoolChicEncoder, CoolChicEncoderParameter
from coolchic.enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
)
from coolchic.enc.utils.parsecli import get_coolchic_param_from_args
from coolchic.hypernet.hypernet import LatentHyperNet
from coolchic.utils.types import HyperNetConfig


class LatentDecoder(CoolChicEncoder):
    """Abstraction over the CoolChicEncoder to use it as a decoder.
    It hides the fact that the CoolChicEncoder stores the latents in the class,
    and allows the user to pass them as arguments.
    """

    def __init__(self, param: CoolChicEncoderParameter):
        super().__init__(param)

    def forward(  # pyright: ignore
        self,
        latents: list[torch.Tensor],
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: torch.Tensor | None = torch.tensor(0.3),
        noise_parameter: torch.Tensor | None = torch.tensor(1.0),
        AC_MAX_VAL: int = -1,
        flag_additional_outputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Replace latents in CoolChicEncoder.
        self.size_per_latent = [(1, *lat.shape[-3:]) for lat in latents]
        self.latent_grids = nn.ParameterList(latents)

        # Forward pass (latents are in the class already).
        return super().forward(
            quantizer_noise_type=quantizer_noise_type,
            quantizer_type=quantizer_type,
            soft_round_temperature=soft_round_temperature,
            noise_parameter=noise_parameter,
            AC_MAX_VAL=AC_MAX_VAL,
            flag_additional_outputs=flag_additional_outputs,
        )

    def get_flops(self) -> None:
        """Changed the forward method's signature, so we need to redefine this method."""
        print("Ignoring get_flops")


class DeltaWholeNet(nn.Module):
    def __init__(self, config: HyperNetConfig):
        super().__init__()
        self.config = config
        coolchic_encoder_parameter = CoolChicEncoderParameter(
            **get_coolchic_param_from_args(config.dec_cfg)
        )
        coolchic_encoder_parameter.set_image_size(config.patch_size)

        self.hypernet = None
        self.encoder = LatentHyperNet(n_latents=self.config.n_latents)
        self.mean_decoder = LatentDecoder(param=coolchic_encoder_parameter)

    def forward(
        self,
        img: torch.Tensor,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "gaussian",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        softround_temperature: float = 0.3,
        noise_parameter: float = 0.25,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        latents = self.encoder.forward(img)

        return self.mean_decoder.forward(
            latents=latents,
            quantizer_noise_type=quantizer_noise_type,
            quantizer_type=quantizer_type,
            soft_round_temperature=torch.tensor(softround_temperature),
            noise_parameter=torch.tensor(noise_parameter),
            AC_MAX_VAL=-1,
            flag_additional_outputs=False,
        )

    def get_mlp_rate(self) -> float:
        # Get MLP rate.
        rate_mlp = 0.0
        rate_per_module = self.mean_decoder.get_network_rate()
        for _, module_rate in rate_per_module.items():  # pyright: ignore
            for _, param_rate in module_rate.items():  # weight, bias
                rate_mlp += param_rate
        return rate_mlp

    def freeze_resnet(self) -> None:
        """Not implemented."""
        pass

    def unfreeze_resnet(self) -> None:
        """Not implemented."""
        pass

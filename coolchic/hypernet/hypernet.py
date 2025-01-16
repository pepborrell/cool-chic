from typing import Any, OrderedDict

import torch
from pydantic import BaseModel
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from coolchic.enc.component.coolchic import CoolChicEncoder, CoolChicEncoderParameter
from coolchic.enc.utils.parsecli import get_coolchic_param_from_args
from coolchic.hypernet.common import ResidualBlockDown, build_mlp
from coolchic.utils.types import HyperNetConfig


class LatentHyperNet(nn.Module):
    def __init__(self, n_latents: int = 7) -> None:
        super().__init__()
        self.n_latents = n_latents

        self.residuals = nn.ModuleList(
            [
                ResidualBlockDown(64, 64, downsample_n=2)
                if i > 0
                else ResidualBlockDown(3, 64, downsample_n=1)
                for i in range(self.n_latents)
            ]
        )
        self.conv1ds = nn.ModuleList(
            [nn.Conv2d(64, 1, kernel_size=1, padding=0) for _ in range(self.n_latents)]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for i in range(self.n_latents):
            x = self.residuals[i](x)
            outputs.append(self.conv1ds[i](x))
        return outputs


def get_backbone(pretrained: bool = True) -> nn.Module:
    if pretrained:
        return resnet50(weights=ResNet50_Weights.DEFAULT)
    return resnet50()


BACKBONE_OUTPUT_FEATURES = 2048


class SynthesisLayerInfo(BaseModel):
    in_ft: int
    out_ft: int
    k_size: int
    mode: str
    non_linearity: str


class SynthesisHyperNet(nn.Module):
    """Takes a latent tensor and outputs the filters of the synthesis network."""

    def __init__(
        self,
        n_latents: int,
        layers_dim: list[str],
        hypernet_hidden_dim: int,
        hypernet_n_layers: int,
    ) -> None:
        super().__init__()
        self.n_input_features = BACKBONE_OUTPUT_FEATURES

        self.n_latents = n_latents
        self.layers_dim = layers_dim
        self.layer_info = self.parse_layers_dim()

        # For hop config, this will be 642 parameters.
        self.n_output_features = self.n_params_synthesis()

        # The layers we need: an MLP with n_layers hidden layers.
        self.hidden_size = hypernet_hidden_dim

        self.mlp = build_mlp(
            input_size=self.n_input_features,
            output_size=self.n_output_features,
            n_hidden_layers=hypernet_n_layers,
            hidden_size=self.hidden_size,
        )

    def parse_layers_dim(self) -> list[SynthesisLayerInfo]:
        """Parses the layers_dim list to a list of SynthesisLayerInfo objects."""
        layer_info = []
        in_ft = self.n_latents  # The input to synthesis are the upsampled latents.
        for layer in self.layers_dim:
            out_ft, k_size, mode, non_linearity = layer.split("-")
            layer_info.append(
                SynthesisLayerInfo(
                    in_ft=in_ft,
                    out_ft=int(out_ft),
                    k_size=int(k_size),
                    mode=mode,
                    non_linearity=non_linearity,
                )
            )
            in_ft = int(out_ft)
        return layer_info

    def n_params_synthesis(self) -> int:
        """Calculates the number of parameters needed for the synthesis network."""
        return sum(
            self.n_params_synthesis_layer(layer_info) for layer_info in self.layer_info
        )

    @staticmethod
    def n_params_synthesis_layer(layer: SynthesisLayerInfo) -> int:
        """Every synthesis layer has out_channels conv kernels of size in_channels x k x k."""
        return layer.in_ft * layer.out_ft * layer.k_size * layer.k_size + layer.out_ft

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.mlp(latent)

    def shape_outputs(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Reshape the output of the hypernetwork to match the shape of the filters."""
        weight_count = 0
        formatted_weights = OrderedDict()
        for layer_num, layer in enumerate(self.layer_info):
            # Select weights for the current layer.
            n_params = self.n_params_synthesis_layer(layer)
            layer_params = x[:, weight_count : weight_count + n_params]
            weight, bias = (
                layer_params[:, : -layer.out_ft],
                layer_params[:, -layer.out_ft :],
            )
            # Reshaping
            weight = weight.view(
                -1, layer.out_ft, layer.in_ft, layer.k_size, layer.k_size
            )
            bias = bias.view(-1, layer.out_ft)
            # Adding to the dictionary
            layer_name = f"layers.{2*layer_num}"  # Multiply by 2 because of the non-linearity layers.
            formatted_weights[f"{layer_name}.weight"] = weight
            formatted_weights[f"{layer_name}.bias"] = bias

            weight_count += n_params

        return formatted_weights


class ArmHyperNet(nn.Module):
    """Takes a latent tensor and outputs the filters of the ARM network."""

    def __init__(
        self,
        dim_arm: int,
        n_hidden_layers: int,
        hypernet_hidden_dim: int,
        hypernet_n_layers: int,
    ) -> None:
        """
        Args:
            dim_arm: Number of context pixels AND dimension of all hidden
                layers.
            n_hidden_layers_arm: Number of hidden layers. Set it to 0 for
                a linear ARM.
        """
        super().__init__()
        self.n_input_features = BACKBONE_OUTPUT_FEATURES

        self.dim_arm = dim_arm
        self.n_hidden_layers = n_hidden_layers
        # For hop config, this will be 544 parameters.
        self.n_output_features = self.n_params_arm()

        # The layers we need: an MLP with hypernet_n_layers hidden layers.
        self.hidden_size = hypernet_hidden_dim

        self.mlp = build_mlp(
            input_size=self.n_input_features,
            output_size=self.n_output_features,
            n_hidden_layers=hypernet_n_layers,
            hidden_size=self.hidden_size,
        )

    def n_params_arm(self) -> int:
        """Calculates the number of parameters needed for the arm network.
        An arm network is an MLP with n_hidden_layers of size dim_arm->dim_arm.
        The output layer outputs 2 values (mu and sigma)."""
        return (
            (self.dim_arm * self.dim_arm + self.dim_arm) * self.n_hidden_layers
            + self.dim_arm * 2
            + 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def shape_outputs(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Reshape the output of the hypernetwork to match the shape of the arm mlp layers."""
        weight_count = 0
        formatted_weights = OrderedDict()
        for layer in range(self.n_hidden_layers):
            # Select weights for the current layer.
            n_params = self.dim_arm * self.dim_arm + self.dim_arm
            layer_params = x[:, weight_count : weight_count + n_params]
            weight, bias = (
                layer_params[:, : -self.dim_arm],
                layer_params[:, -self.dim_arm :],
            )
            # Reshaping
            weight = weight.view(-1, self.dim_arm, self.dim_arm)
            bias = bias.view(-1, self.dim_arm)
            weight_count += n_params

            # Adding to the dictionary
            layer_name = f"mlp.{2*layer}"  # Multiplying by 2 because we have activations interleaved.
            formatted_weights[f"{layer_name}.weight"] = weight
            formatted_weights[f"{layer_name}.bias"] = bias

            weight_count += n_params

        # Output layer
        n_params = self.dim_arm * 2 + 2
        layer_params = x[:, weight_count : weight_count + n_params]
        weight, bias = (
            layer_params[:, :-2],
            layer_params[:, -2:],
        )
        formatted_weights[f"mlp.{2*self.n_hidden_layers}.weight"] = weight
        formatted_weights[f"mlp.{2*self.n_hidden_layers}.bias"] = bias

        return formatted_weights


class UpsamplingHyperNet(nn.Module):
    """Takes a latent tensor and outputs the filters of the upsampling network.
    The upsampling network has L-1 convtransposes and L-1 pre-concat filters
    (where L is the number of latents). So we output those.
    """

    def __init__(
        self,
        n_latents: int,
        ups_k_size: int,
        ups_preconcat_k_size: int,
        hypernet_hidden_dim: int,
        hypernet_n_layers: int,
    ):
        super().__init__()
        self.n_input_features = BACKBONE_OUTPUT_FEATURES

        self.ups_k_size = ups_k_size
        self.ups_n_params = self.symmetric_filter_n_params_from_target(ups_k_size)
        self.ups_preconcat_k_size = ups_preconcat_k_size
        self.ups_preconcat_n_params = self.symmetric_filter_n_params_from_target(
            ups_preconcat_k_size
        )
        self.n_latents = n_latents

        # Calculations:
        # 1. The ConvTranspose2ds are separable symmetric filters,
        #     so we only need ups_n_params parameters for each, plus one bias.
        # 2. The pre-concat filters are separable symmetric convolutions,
        #     so we only need ups_preconcat_n_params parameters for each, plus the bias.
        self.n_output_features = (self.n_latents - 1) * (
            self.ups_n_params + 1 + self.ups_preconcat_n_params + 1
        )

        # The layers we need: an MLP with hypernet_n_layers hidden layers.
        self.hidden_size = hypernet_hidden_dim
        self.mlp = build_mlp(
            input_size=self.n_input_features,
            output_size=self.n_output_features,
            n_hidden_layers=hypernet_n_layers,
            hidden_size=self.hidden_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    @staticmethod
    def symmetric_filter_n_params_from_target(target_size: int) -> int:
        """Calculates the number of parameters needed for a symmetric filter
        of size target_size.

        For a kernel of size target_k_size = 2N, we need N values
        e.g. 3 params a b c to parameterize a b c c b a.
        For a kernel of size target_k_size = 2N + 1, we need N + 1 values
        e.g. 4 params a b c d to parameterize a b c d c b a.
        """
        return (target_size + 1) // 2

    def shape_output(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Reshape the output of the hypernetwork to match the shape of the upsampling filters."""
        weight_count = 0
        formatted_weights = OrderedDict()
        for n_stage in range(self.n_latents - 1):
            # Select weights for the current layer.
            n_params_transpose = self.ups_n_params + 1
            n_params_preconcat = self.ups_preconcat_n_params + 1
            stage_params = x[
                :, weight_count : weight_count + n_params_transpose + n_params_preconcat
            ]
            transpose_params, preconcat_params = (
                stage_params[:, :n_params_transpose],
                stage_params[:, n_params_transpose:],
            )

            # First we do the transpose filters.
            transpose_weight, transpose_bias = (
                transpose_params[:, :-1],
                transpose_params[:, -1],
            )
            # Reshaping
            transpose_weight = transpose_weight.view(-1, self.ups_n_params)
            transpose_bias = transpose_bias.view(-1, 1)
            formatted_weights[f"conv_transpose2ds.{n_stage}.bias"] = transpose_bias
            formatted_weights[
                f"conv_transpose2ds.{n_stage}.parametrizations.weight.original"
            ] = transpose_weight

            # Now we do the preconcat filters.
            preconcat_weight, preconcat_bias = (
                preconcat_params[:, :-1],
                preconcat_params[:, -1],
            )
            # Reshaping
            preconcat_weight = preconcat_weight.view(-1, self.ups_preconcat_n_params)
            preconcat_bias = preconcat_bias.view(-1, 1)
            formatted_weights[f"conv2ds.{n_stage}.bias"] = preconcat_bias
            formatted_weights[f"conv2ds.{n_stage}.parametrizations.weight.original"] = (
                preconcat_weight
            )

            weight_count += n_params_transpose + n_params_preconcat

        return formatted_weights


class CoolchicHyperNet(nn.Module):
    def __init__(self, config: HyperNetConfig) -> None:
        super().__init__()
        self.config = config

        # Instantiate all the hypernetworks.
        self.latent_hn = LatentHyperNet(n_latents=self.config.n_latents)
        self.hn_backbone = get_backbone(pretrained=True)
        self.synthesis_hn = SynthesisHyperNet(
            n_latents=self.config.n_latents,
            layers_dim=self.config.dec_cfg.parsed_layers_synthesis,
            hypernet_hidden_dim=self.config.synthesis.hidden_dim,
            hypernet_n_layers=self.config.synthesis.n_layers,
        )
        self.arm_hn = ArmHyperNet(
            dim_arm=self.config.dec_cfg.dim_arm,
            n_hidden_layers=self.config.dec_cfg.n_hidden_layers_arm,
            hypernet_hidden_dim=self.config.arm.hidden_dim,
            hypernet_n_layers=self.config.arm.n_layers,
        )
        self.upsampling_hn = UpsamplingHyperNet(
            n_latents=self.config.n_latents,
            ups_k_size=self.config.dec_cfg.ups_k_size,
            ups_preconcat_k_size=self.config.dec_cfg.ups_preconcat_k_size,
            hypernet_hidden_dim=self.config.upsampling.hidden_dim,
            hypernet_n_layers=self.config.upsampling.n_layers,
        )

    def forward(
        self, img: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """This strings together all hypernetwork components."""
        latent_weights = self.latent_hn(img)
        features = self.hn_backbone(latent_weights)
        synthesis_weights = self.synthesis_hn(features)
        arm_weights = self.arm_hn(features)
        upsampling_weights = self.upsampling_hn(features)
        return (
            latent_weights,
            synthesis_weights,
            arm_weights,
            upsampling_weights,
        )


class CoolchicWholeNet(nn.Module):
    def __init__(self, config: HyperNetConfig):
        super().__init__()
        self.config = config
        coolchic_encoder_parameter = CoolChicEncoderParameter(
            **get_coolchic_param_from_args(config.dec_cfg)
        )

        self.hypernet = CoolchicHyperNet(config)
        self.cc_encoder = CoolChicEncoder(param=coolchic_encoder_parameter)

    def forward(
        self, img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        latent_weights, synthesis_weights, arm_weights, upsampling_weights = (
            self.hypernet(img)
        )

        # Replacing weights.
        self.cc_encoder.latent_grids = latent_weights
        self.cc_encoder.synthesis.set_hypernet_weights(synthesis_weights)
        self.cc_encoder.arm.set_hypernet_weights(arm_weights)
        self.cc_encoder.upsampling.set_hypernet_weights(upsampling_weights)

        # TODO: get these parameters from input.
        return self.cc_encoder(
            quantizer_noise_type="kumaraswamy",
            quantizer_type="softround",
            soft_round_temperature=0.3,
            noise_parameter=1.0,
            AC_MAX_VAL=-1,
            flag_additional_outputs=False,
        )

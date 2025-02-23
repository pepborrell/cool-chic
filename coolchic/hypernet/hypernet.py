from typing import Any, Literal, OrderedDict

import torch
from pydantic import BaseModel
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from coolchic.enc.component.coolchic import CoolChicEncoder, CoolChicEncoderParameter
from coolchic.enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
)
from coolchic.enc.utils.parsecli import get_coolchic_param_from_args
from coolchic.hypernet.common import ResidualBlockDown, build_mlp, upsample_latents
from coolchic.utils.nn import get_num_of_params
from coolchic.utils.types import HyperNetConfig


class LatentHyperNet(nn.Module):
    def __init__(self, n_latents: int = 7) -> None:
        super().__init__()
        self.n_latents = n_latents

        self.residual_blocks = nn.ModuleList(
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
            x = self.residual_blocks[i](x)
            outputs.append(self.conv1ds[i](x))
        return outputs


def get_backbone(
    pretrained: bool = True,
    arch: Literal["resnet18", "resnet50"] = "resnet18",
    input_channels: int = 3,
) -> tuple[nn.Module, int]:
    if arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        n_output_features = 512
    elif arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        n_output_features = 2048

    if input_channels != 3:
        # Replace the first layer with a new one. For cases where the input is not RGB images.
        model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    # We want to extract the features, so we remove the final fc layer.
    model = torch.nn.Sequential(*list(model.children())[:-1], nn.Flatten(start_dim=1))

    return model, n_output_features


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
        n_input_features: int,
        hypernet_hidden_dim: int,
        hypernet_n_layers: int,
    ) -> None:
        super().__init__()
        self.n_input_features = n_input_features

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
            output_activation=nn.Tanh(),
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
            weight = weight.view(layer.out_ft, layer.in_ft, layer.k_size, layer.k_size)
            bias = bias.view(layer.out_ft)
            # Adding to the dictionary
            layer_name = f"layers.{2 * layer_num}"  # Multiply by 2 because of the non-linearity layers.
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
        n_input_features: int,
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
        self.n_input_features = n_input_features

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
            output_activation=nn.Tanh(),
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
        # Batch dimension shouldn't be there when passing to the model.
        x = x.squeeze(0)

        weight_count = 0
        formatted_weights = OrderedDict()
        for layer in range(self.n_hidden_layers):
            # Select weights for the current layer.
            n_params = self.dim_arm * self.dim_arm + self.dim_arm
            layer_params = x[weight_count : weight_count + n_params]
            weight, bias = (
                layer_params[: -self.dim_arm],
                layer_params[-self.dim_arm :],
            )
            # Reshaping
            weight = weight.view(self.dim_arm, self.dim_arm)
            bias = bias.view(self.dim_arm)

            # Adding to the dictionary
            layer_name = f"mlp.{2 * layer}"  # Multiplying by 2 because we have activations interleaved.
            formatted_weights[f"{layer_name}.weight"] = weight
            formatted_weights[f"{layer_name}.bias"] = bias

            weight_count += n_params

        # Output layer
        n_params = self.dim_arm * 2 + 2
        layer_params = x[weight_count : weight_count + n_params]
        weight, bias = (
            layer_params[:-2],
            layer_params[-2:],
        )
        # Reshaping
        weight = weight.view(2, self.dim_arm)
        bias = bias.view(2)

        formatted_weights[f"mlp.{2 * self.n_hidden_layers}.weight"] = weight
        formatted_weights[f"mlp.{2 * self.n_hidden_layers}.bias"] = bias

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
        n_input_features: int,
        hypernet_hidden_dim: int,
        hypernet_n_layers: int,
    ):
        super().__init__()
        self.n_input_features = n_input_features

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
            output_activation=nn.Tanh(),
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
            transpose_weight = transpose_weight.view(self.ups_n_params)
            transpose_bias = transpose_bias.view(1)
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
            preconcat_weight = preconcat_weight.view(self.ups_preconcat_n_params)
            preconcat_bias = preconcat_bias.view(1)
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
        self.hn_backbone, backbone_n_features = get_backbone(
            pretrained=True, arch=config.backbone_arch
        )
        self.latent_backbone, _ = get_backbone(
            pretrained=True,
            arch=config.backbone_arch,
            input_channels=self.config.n_latents,
        )
        if self.config.lmbda_as_feature:
            backbone_n_features += 1

        self.synthesis_hn = SynthesisHyperNet(
            n_latents=self.config.n_latents,
            layers_dim=self.config.dec_cfg.parsed_layers_synthesis,
            n_input_features=2 * backbone_n_features,
            hypernet_hidden_dim=self.config.synthesis.hidden_dim,
            hypernet_n_layers=self.config.synthesis.n_layers,
        )
        self.arm_hn = ArmHyperNet(
            dim_arm=self.config.dec_cfg.dim_arm,
            n_hidden_layers=self.config.dec_cfg.n_hidden_layers_arm,
            n_input_features=2 * backbone_n_features,
            hypernet_hidden_dim=self.config.arm.hidden_dim,
            hypernet_n_layers=self.config.arm.n_layers,
        )

        self.print_n_params_submodule()

    def forward(
        self, img: torch.Tensor, lmbda: torch.Tensor | None = None
    ) -> tuple[
        list[torch.Tensor],
        OrderedDict[str, torch.Tensor],
        OrderedDict[str, torch.Tensor],
    ]:
        """This strings together all hypernetwork components."""
        latent_weights = self.latent_hn.forward(img)
        img_features = self.hn_backbone.forward(img)
        latent_features = self.latent_backbone.forward(
            upsample_latents(latent_weights, mode="bicubic").detach()
        )
        features = torch.cat([img_features, latent_features], dim=1)
        if self.config.lmbda_as_feature:
            assert lmbda is not None
            features = torch.cat([features, lmbda.unsqueeze(0)], dim=1)
        synthesis_weights = self.synthesis_hn.forward(features)
        arm_weights = self.arm_hn.forward(features)

        return (
            latent_weights,
            self.synthesis_hn.shape_outputs(synthesis_weights),
            self.arm_hn.shape_outputs(arm_weights),
        )

    def print_n_params_submodule(self):
        total_params = get_num_of_params(self)

        def format_param_str(subm_name: str, n_params: int) -> str:
            return f"{subm_name}: {n_params}, {100 * n_params / total_params:.2f}%\n"

        output_str = (
            f"NUMBER OF PARAMETERS:\nTotal number of parameters: {total_params}\n"
        )

        output_str += format_param_str("latent", get_num_of_params(self.latent_hn))
        output_str += format_param_str("backbone", get_num_of_params(self.hn_backbone))
        output_str += format_param_str(
            "synthesis", get_num_of_params(self.synthesis_hn)
        )
        output_str += format_param_str("arm", get_num_of_params(self.arm_hn))
        print(output_str)


class CoolchicWholeNet(nn.Module):
    def __init__(self, config: HyperNetConfig):
        super().__init__()
        self.config = config
        coolchic_encoder_parameter = CoolChicEncoderParameter(
            **get_coolchic_param_from_args(config.dec_cfg)
        )
        coolchic_encoder_parameter.set_image_size(config.patch_size)

        self.hypernet = CoolchicHyperNet(config)
        self.cc_encoder = CoolChicEncoder(param=coolchic_encoder_parameter)

    def image_to_coolchic(
        self,
        img: torch.Tensor,
        stop_grads: bool = False,
        lmbda: torch.Tensor | None = None,
    ) -> CoolChicEncoder:
        latent_weights, synthesis_weights, arm_weights = self.hypernet.forward(
            img, lmbda=lmbda
        )
        # Make them leaves in the graph.
        if stop_grads:
            latent_weights = [nn.Parameter(lat) for lat in latent_weights]
            synthesis_weights = OrderedDict(
                {k: nn.Parameter(v) for k, v in synthesis_weights.items()}
            )
            arm_weights = OrderedDict(
                {k: nn.Parameter(v) for k, v in arm_weights.items()}
            )

        # Replacing weights.
        self.cc_encoder.synthesis.set_hypernet_weights(synthesis_weights)
        self.cc_encoder.arm.set_hypernet_weights(arm_weights)
        # Set upsampling as a bicubic upsampling filter.
        self.cc_encoder.upsampling.reinitialize_parameters()

        # Replace latents, they are not exactly like a module's weights.
        self.cc_encoder.size_per_latent = [
            (1, *lat.shape[-3:]) for lat in latent_weights
        ]
        self.cc_encoder.latent_grids = nn.ParameterList(latent_weights)

        return self.cc_encoder

    def forward(
        self,
        img: torch.Tensor,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "gaussian",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        softround_temperature: float = 0.3,
        noise_parameter: float = 0.25,
        lmbda: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return self.image_to_coolchic(img, lmbda=lmbda).forward(
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
        rate_per_module = self.cc_encoder.get_network_rate()
        for _, module_rate in rate_per_module.items():  # pyright: ignore
            for _, param_rate in module_rate.items():  # weight, bias
                rate_mlp += param_rate
        return rate_mlp

    def freeze_resnet(self):
        for param in self.hypernet.hn_backbone.parameters():
            param.requires_grad = False

    def unfreeze_resnet(self):
        for param in self.hypernet.hn_backbone.parameters():
            param.requires_grad = True

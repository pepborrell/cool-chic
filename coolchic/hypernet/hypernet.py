import abc
from typing import Any, Literal, OrderedDict

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from pydantic import BaseModel
from torch import nn
from torch.func import functional_call
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from coolchic.enc.component.coolchic import CoolChicEncoder, CoolChicEncoderParameter
from coolchic.enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
)
from coolchic.enc.component.nlcoolchic import LatentFreeCoolChicEncoder
from coolchic.enc.utils.parsecli import get_coolchic_param_from_args
from coolchic.hypernet.common import ResidualBlockDown, build_mlp, upsample_latents
from coolchic.utils.nn import get_num_of_params
from coolchic.utils.types import HyperNetConfig


class LatentHyperNet(nn.Module):
    def __init__(self, n_latents: int = 7, n_hidden_channels: int = 64) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.n_hidden_channels = n_hidden_channels

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlockDown(
                    self.n_hidden_channels, self.n_hidden_channels, downsample_n=2
                )
                if i > 0
                else ResidualBlockDown(3, self.n_hidden_channels, downsample_n=1)
                for i in range(self.n_latents)
            ]
        )
        self.conv1ds = nn.ModuleList(
            [
                nn.Conv2d(self.n_hidden_channels, 1, kernel_size=1, padding=0)
                for _ in range(self.n_latents)
            ]
        )

        # For flop analysis.
        self.flops_per_module = {k: 0 for k in ["residual_blocks", "conv1ds"]}

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for i in range(self.n_latents):
            x = self.residual_blocks[i](x)
            outputs.append(self.conv1ds[i](x))
        return outputs

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, ResidualBlockDown):
            m.reset_weights()
        elif isinstance(m, nn.Conv2d):
            # Initialize the conv layers with a normal distribution.
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def reinitialize_parameters(self) -> None:
        self.apply(self._init_weights)

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

        mock_img_size = (1, 3, 512, 512)
        flops = FlopCountAnalysis(
            self,
            torch.zeros(mock_img_size),  # img
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

    # Strange aux method for flop analysis.
    def get_backbone_flops(self) -> int:
        # Count the number of floating point operations here. It must be done before
        # torch scripting the different modules.

        self = self.train(mode=False)

        mock_img_size = (1, 3, 512, 512)
        flops = FlopCountAnalysis(
            self,
            torch.zeros(mock_img_size),  # img
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

    import types

    model.flops_per_module = {k: 0 for k, _ in model.named_modules()}  # pyright: ignore
    model.get_flops = types.MethodType(get_backbone_flops, model)  # pyright: ignore

    return model, n_output_features


class SynthesisLayerInfo(BaseModel):
    in_ft: int
    out_ft: int
    k_size: int
    mode: str
    non_linearity: str
    bias: bool


class SynthesisHyperNet(nn.Module):
    """Takes a latent tensor and outputs the filters of the synthesis network."""

    def __init__(
        self,
        n_latents: int,
        layers_dim: list[str],
        n_input_features: int,
        hypernet_hidden_dim: int,
        hypernet_n_layers: int,
        biases: bool,
        only_biases: bool = False,
    ) -> None:
        """
        Args:
            n_latents: Number of latents.
            layers_dim: List of strings with the format "out_ft-k_size-mode-non_linearity".
                The mode can be "conv" or "conv_transpose". The non_linearity can be
                "relu", "leaky_relu", "tanh", or "sigmoid".
            n_input_features: Number of input features to the hypernetwork.
            hypernet_hidden_dim: Hidden dimension of the hypernetwork.
            hypernet_n_layers: Number of hidden layers in the hypernetwork.
            biases: Whether to use biases in the synthesis network.
            only_biases: If True, only output biases, similar to what is used in COIN++.
        """
        super().__init__()
        self.n_input_features = n_input_features

        self.n_latents = n_latents
        self.layers_dim = layers_dim
        self.biases = biases
        self.only_biases = only_biases
        self.layer_info = self.parse_layers_dim(biases=self.biases)

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

        # For flop analysis.
        self.flops_per_module = {k: 0 for k in ["mlp"]}

    def parse_layers_dim(self, biases: bool) -> list[SynthesisLayerInfo]:
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
                    bias=biases,
                )
            )
            in_ft = int(out_ft)
        return layer_info

    def n_params_synthesis(self) -> int:
        """Calculates the number of parameters needed for the synthesis network."""
        return sum(
            self.n_params_synthesis_layer(layer_info, self.only_biases)
            for layer_info in self.layer_info
        )

    @staticmethod
    def n_params_synthesis_layer(layer: SynthesisLayerInfo, only_biases: bool) -> int:
        """Every synthesis layer has out_channels conv kernels of size in_channels x k x k."""
        if only_biases:
            return layer.out_ft
        n_params_kernels = layer.in_ft * layer.out_ft * layer.k_size * layer.k_size
        n_params_bias = layer.out_ft if layer.bias else 0
        return n_params_kernels + n_params_bias

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.mlp(latent)

    def shape_outputs(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Reshape the output of the hypernetwork to match the shape of the filters."""
        weight_count = 0
        formatted_weights = OrderedDict()
        for layer_num, layer in enumerate(self.layer_info):
            # Select weights for the current layer.
            n_params = self.n_params_synthesis_layer(layer, self.only_biases)
            layer_params = x[:, weight_count : weight_count + n_params]
            weight_count += n_params
            layer_name = f"layers.{2 * layer_num}"  # Multiply by 2 because of the non-linearity layers.

            # Adding weight.
            if self.only_biases:
                # Only biases, no weights.
                weight = None
                bias = layer_params.view(-1, layer.out_ft)  # Keeping a batch dimension.
            else:
                weight = layer_params[
                    :, : layer.out_ft * layer.in_ft * layer.k_size * layer.k_size
                ].view(-1, layer.out_ft, layer.in_ft, layer.k_size, layer.k_size)
                bias = None
                # Adding bias if needed.
                if layer.bias:
                    bias = layer_params[:, -layer.out_ft :].view(-1, layer.out_ft)

            # Adding to the dictionary
            if weight is not None:
                formatted_weights[f"{layer_name}.weight"] = weight
            if bias is not None:
                formatted_weights[f"{layer_name}.bias"] = bias

        return formatted_weights

    def get_flops(self) -> int:
        # Count the number of floating point operations here. It must be done before
        # torch scripting the different modules.

        self = self.train(mode=False)

        flops = FlopCountAnalysis(
            self,
            torch.zeros(1, self.n_input_features),  # output of backbone.
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


class ArmHyperNet(nn.Module):
    """Takes a latent tensor and outputs the filters of the ARM network."""

    def __init__(
        self,
        dim_arm: int,
        n_hidden_layers: int,
        n_input_features: int,
        hypernet_hidden_dim: int,
        hypernet_n_layers: int,
        biases: bool,
        only_biases: bool = False,
    ) -> None:
        """
        Args:
            dim_arm: Number of context pixels AND dimension of all hidden
                layers.
            n_hidden_layers_arm: Number of hidden layers. Set it to 0 for
                a linear ARM.
            n_input_features: Number of input features to the hypernetwork.
            hypernet_hidden_dim: Hidden dimension of the hypernetwork.
            hypernet_n_layers: Number of hidden layers in the hypernetwork.
            biases: Whether to use biases in the synthesis network.
            only_biases: If True, only output biases, similar to what is used in COIN++.
        """
        super().__init__()
        self.n_input_features = n_input_features

        self.dim_arm = dim_arm
        self.n_hidden_layers = n_hidden_layers
        self.biases = biases
        self.only_biases = only_biases
        # For hop config, this will be 544 parameters.
        self.n_output_features = self.n_params_arm(
            biases=self.biases, only_biases=self.only_biases
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

        # For flop analysis.
        self.flops_per_module = {k: 0 for k in ["mlp"]}

    def n_params_arm(self, biases: bool, only_biases: bool) -> int:
        """Calculates the number of parameters needed for the arm network.
        An arm network is an MLP with n_hidden_layers of size dim_arm->dim_arm.
        The output layer outputs 2 values (mu and sigma)."""
        if only_biases:
            return self.n_hidden_layers * self.dim_arm + 2

        n_params_interm_layer = (
            self.dim_arm * self.dim_arm + self.dim_arm
            if biases
            else self.dim_arm * self.dim_arm
        )
        bias_outp_layer = 2 if biases else 0

        return (
            n_params_interm_layer * self.n_hidden_layers
            # Output layer.
            + self.dim_arm * 2
            + bias_outp_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def shape_outputs(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Reshape the output of the hypernetwork to match the shape of the arm mlp layers."""
        weight_count = 0
        formatted_weights = OrderedDict()
        for layer in range(self.n_hidden_layers):
            layer_name = f"mlp.{2 * layer}"  # Multiplying by 2 because we have activations interleaved.
            if self.only_biases:
                # Only biases, no weights.
                n_params = self.dim_arm
                weight = None
                bias = x[:, weight_count : weight_count + n_params].view(
                    -1, self.dim_arm
                )  # Keeping a batch dimension.
            else:
                n_params = self.dim_arm * self.dim_arm
                n_params += self.dim_arm if self.biases else 0
                # Select weights for the current layer.
                layer_params = x[:, weight_count : weight_count + n_params]

                # Adding weight.
                weight = layer_params[:, : self.dim_arm**2].view(
                    -1, self.dim_arm, self.dim_arm
                )
                bias = None
                # Adding bias, if needed.
                if self.biases:
                    bias = layer_params[:, self.dim_arm**2 :].view(-1, self.dim_arm)
            # Adding to the dictionary
            if weight is not None:
                formatted_weights[f"{layer_name}.weight"] = weight
            if bias is not None:
                formatted_weights[f"{layer_name}.bias"] = bias

            weight_count += n_params

        # Output layer
        if self.only_biases:
            # Only biases, no weights.
            n_params = 2
            weight = None
            bias = x[:, weight_count : weight_count + n_params].view(-1, 2)
            formatted_weights[f"mlp.{2 * self.n_hidden_layers}.bias"] = bias
        else:
            n_params = self.dim_arm * 2
            n_params += 2 if self.biases else 0
            layer_params = x[:, weight_count : weight_count + n_params]
            # Adding weight.
            weight = layer_params[:, : self.dim_arm * 2].view(-1, 2, self.dim_arm)
            formatted_weights[f"mlp.{2 * self.n_hidden_layers}.weight"] = weight
            # Adding bias, if needed.
            if self.biases:
                bias = layer_params[:, self.dim_arm * 2 :].view(-1, 2)
                formatted_weights[f"mlp.{2 * self.n_hidden_layers}.bias"] = bias

        return formatted_weights

    def get_flops(self) -> int:
        # Count the number of floating point operations here. It must be done before
        # torch scripting the different modules.

        self = self.train(mode=False)

        flops = FlopCountAnalysis(
            self,
            torch.zeros(1, self.n_input_features),  # output of backbone.
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

    def shape_outputs(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
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
        self.hn_backbone, backbone_n_features = get_backbone(
            pretrained=True, arch=config.backbone_arch
        )
        # Commented out because we are not using the latent backbone.
        # self.latent_backbone, _ = get_backbone(
        #     pretrained=True,
        #     arch=config.backbone_arch,
        #     input_channels=self.config.n_latents,
        # )

        self.synthesis_hn = SynthesisHyperNet(
            n_latents=self.config.n_latents,
            layers_dim=self.config.dec_cfg.parsed_layers_synthesis,
            # n_input_features=2 * backbone_n_features,
            n_input_features=backbone_n_features,
            hypernet_hidden_dim=self.config.synthesis.hidden_dim,
            hypernet_n_layers=self.config.synthesis.n_layers,
            biases=self.config.synthesis.biases,
        )
        self.arm_hn = ArmHyperNet(
            dim_arm=self.config.dec_cfg.dim_arm,
            n_hidden_layers=self.config.dec_cfg.n_hidden_layers_arm,
            # n_input_features=2 * backbone_n_features,
            n_input_features=backbone_n_features,
            hypernet_hidden_dim=self.config.arm.hidden_dim,
            hypernet_n_layers=self.config.arm.n_layers,
            biases=self.config.arm.biases,
        )

    def forward(
        self, img: torch.Tensor
    ) -> tuple[
        list[torch.Tensor],
        OrderedDict[str, torch.Tensor],
        OrderedDict[str, torch.Tensor],
    ]:
        """This strings together all hypernetwork components."""
        latent_weights = self.latent_hn.forward(img)
        img_features = self.hn_backbone.forward(img)
        # Commented out because we are not using the latent backbone.
        # latent_features = self.latent_backbone.forward(
        #     upsample_latents(
        #         latent_weights,
        #         mode="bicubic",
        #         img_size=(img.shape[-2], img.shape[-1]),
        #     ).detach()
        # )
        # features = torch.cat([img_features, latent_features], dim=1)
        features = img_features
        synthesis_weights = self.synthesis_hn.forward(features)
        arm_weights = self.arm_hn.forward(features)

        return (
            latent_weights,
            self.synthesis_hn.shape_outputs(synthesis_weights),
            self.arm_hn.shape_outputs(arm_weights),
        )

    def latent_forward(self, img: torch.Tensor) -> list[torch.Tensor]:
        """This strings together all hypernetwork components."""
        latent_weights = self.latent_hn.forward(img)
        return latent_weights

    def print_n_params_submodule(self):
        total_params = get_num_of_params(self)

        def format_param_str(subm_name: str, n_params: int) -> str:
            return f"{subm_name}: {n_params}, {100 * n_params / total_params:.2f}%\n"

        output_str = (
            "NUMBER OF (TRAINABLE) PARAMETERS:\n"
            f"Total number of parameters: {total_params}\n"
        )

        output_str += format_param_str("latent", get_num_of_params(self.latent_hn))
        output_str += format_param_str("backbone", get_num_of_params(self.hn_backbone))
        output_str += format_param_str(
            "synthesis", get_num_of_params(self.synthesis_hn)
        )
        output_str += format_param_str("arm", get_num_of_params(self.arm_hn))
        print(output_str)

    def init_deltas(self) -> None:
        """Initialize the hypernetwork weights knowing that they will be used as deltas.
        Because we usually start from a pretrained model, we want the deltas that
        are outputted in the beggining to be zero.
        """

        # zero out deltas, so at the start we have N-O coolchic.
        def zero_out_last_layer(mod: nn.Module) -> None:
            layers = [m for m in mod.modules() if isinstance(m, nn.Linear)]
            last_layer = layers[-1]  # The last linear layer

            # Zero out its parameters
            with torch.no_grad():
                last_layer.weight.fill_(0)
                last_layer.bias.fill_(0)

        zero_out_last_layer(self.synthesis_hn.mlp)
        zero_out_last_layer(self.arm_hn.mlp)


class SmallCoolchicHyperNet(CoolchicHyperNet):
    def __init__(self, config: HyperNetConfig) -> None:
        nn.Module.__init__(self)
        self.config = config

        # Instantiate all the hypernetworks.
        self.latent_hn = LatentHyperNet(n_latents=self.config.n_latents)

        # One whole conv backbone, simple.
        n_input_channels = 3 + self.config.n_latents
        self.backbone = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            # To get a fixed size representation and flatten it, use average pooling.
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )
        self.backbone_n_features = 1024

        self.synthesis_hn = SynthesisHyperNet(
            n_latents=self.config.n_latents,
            layers_dim=self.config.dec_cfg.parsed_layers_synthesis,
            n_input_features=self.backbone_n_features,
            hypernet_hidden_dim=self.config.synthesis.hidden_dim,
            hypernet_n_layers=self.config.synthesis.n_layers,
            biases=self.config.synthesis.biases,
        )
        self.arm_hn = ArmHyperNet(
            dim_arm=self.config.dec_cfg.dim_arm,
            n_hidden_layers=self.config.dec_cfg.n_hidden_layers_arm,
            n_input_features=self.backbone_n_features,
            hypernet_hidden_dim=self.config.arm.hidden_dim,
            hypernet_n_layers=self.config.arm.n_layers,
            biases=self.config.arm.biases,
        )

    def forward(
        self, img: torch.Tensor
    ) -> tuple[
        list[torch.Tensor],
        OrderedDict[str, torch.Tensor],
        OrderedDict[str, torch.Tensor],
    ]:
        """This strings together all hypernetwork components."""
        latent_weights = self.latent_hn.forward(img)

        # Get features from backbone.
        upsampled_latents = upsample_latents(
            latent_weights,
            mode="bicubic",
            img_size=(img.shape[-2], img.shape[-1]),
        ).detach()
        features = self.backbone.forward(torch.cat([img, upsampled_latents], dim=1))

        # Use features to get synthesis and arm weights.
        synthesis_weights = self.synthesis_hn.forward(features)
        arm_weights = self.arm_hn.forward(features)

        return (
            latent_weights,
            self.synthesis_hn.shape_outputs(synthesis_weights),
            self.arm_hn.shape_outputs(arm_weights),
        )

    def print_n_params_submodule(self):
        """Had to change this method because we changed some names."""
        total_params = get_num_of_params(self)

        def format_param_str(subm_name: str, n_params: int) -> str:
            return f"{subm_name}: {n_params}, {100 * n_params / total_params:.2f}%\n"

        output_str = (
            "NUMBER OF (TRAINABLE) PARAMETERS:\n"
            f"Total number of parameters: {total_params}\n"
        )

        output_str += format_param_str("latent", get_num_of_params(self.latent_hn))
        output_str += format_param_str("backbone", get_num_of_params(self.backbone))
        output_str += format_param_str(
            "synthesis", get_num_of_params(self.synthesis_hn)
        )
        output_str += format_param_str("arm", get_num_of_params(self.arm_hn))
        print(output_str)


class SmallAdditiveHyperNet(SmallCoolchicHyperNet):
    def __init__(self, config: HyperNetConfig) -> None:
        super().__init__(config)

        # Redefine the prediction heads so that they only output biases.
        self.synthesis_hn = SynthesisHyperNet(
            n_latents=self.config.n_latents,
            layers_dim=self.config.dec_cfg.parsed_layers_synthesis,
            n_input_features=self.backbone_n_features,
            hypernet_hidden_dim=self.config.synthesis.hidden_dim,
            hypernet_n_layers=self.config.synthesis.n_layers,
            biases=self.config.synthesis.biases,
            only_biases=True,
        )
        self.arm_hn = ArmHyperNet(
            dim_arm=self.config.dec_cfg.dim_arm,
            n_hidden_layers=self.config.dec_cfg.n_hidden_layers_arm,
            n_input_features=self.backbone_n_features,
            hypernet_hidden_dim=self.config.arm.hidden_dim,
            hypernet_n_layers=self.config.arm.n_layers,
            biases=self.config.arm.biases,
            only_biases=True,
        )

        self.print_n_params_submodule()


# Abstract WholeNet class, to indicate that the class is a whole network.
class WholeNet(nn.Module, abc.ABC):
    config: HyperNetConfig

    @abc.abstractmethod
    def forward(
        self,
        img: torch.Tensor,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "gaussian",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        softround_temperature: torch.Tensor = torch.tensor(0.3),
        noise_parameter: torch.Tensor = torch.tensor(0.25),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        pass

    @abc.abstractmethod
    def image_to_coolchic(self, img: torch.Tensor, stop_grads: bool) -> CoolChicEncoder:
        pass

    @abc.abstractmethod
    def get_mlp_rate(self) -> float:
        pass

    @abc.abstractmethod
    def freeze_resnet(self):
        pass

    @abc.abstractmethod
    def unfreeze_resnet(self):
        pass


class CoolchicWholeNet(WholeNet):
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
        self, img: torch.Tensor, stop_grads: bool = False
    ) -> CoolChicEncoder:
        img = img.to(self.hypernet.latent_hn.conv1ds[0].weight.device)
        latent_weights, synthesis_weights, arm_weights = self.hypernet.forward(img)
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
        self.cc_encoder.size_per_latent = [lat.shape for lat in latent_weights]
        # Something like self.cc_encoder.latent_grids = nn.ParameterList(latent_weights)
        # would break the computation graph. This doesn't. Following tips in:
        # https://github.com/qu-gg/torch-hypernetwork-tutorials?tab=readme-ov-file#tensorVSparameter
        for i in range(len(self.cc_encoder.latent_grids)):
            del self.cc_encoder.latent_grids[i].data
            self.cc_encoder.latent_grids[i].data = latent_weights[i]

        return self.cc_encoder

    def forward(
        self,
        img: torch.Tensor,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "gaussian",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        softround_temperature: torch.Tensor = torch.tensor(0.3),
        noise_parameter: torch.Tensor = torch.tensor(0.25),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return self.image_to_coolchic(img).forward(
            quantizer_noise_type=quantizer_noise_type,
            quantizer_type=quantizer_type,
            soft_round_temperature=softround_temperature,
            noise_parameter=noise_parameter,
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
        for name, param in self.hypernet.latent_backbone.named_parameters():
            if "conv1" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_resnet(self):
        for param in self.hypernet.hn_backbone.parameters():
            param.requires_grad = True
        for param in self.hypernet.latent_backbone.parameters():
            param.requires_grad = True


class LatentDecoder(CoolChicEncoder):
    """Abstraction over the CoolChicEncoder to use it as a decoder.
    It hides the fact that the CoolChicEncoder stores the latents in the class,
    and allows the user to pass them as arguments.
    """

    def __init__(
        self, param: CoolChicEncoderParameter, only_delta_biases: bool = False
    ):
        super().__init__(param)
        self.param = param
        self.only_delta_biases = only_delta_biases

    def forward(  # pyright: ignore
        self,
        latents: list[torch.Tensor],
        synth_delta: list[torch.Tensor] | None,
        arm_delta: list[torch.Tensor] | None,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: torch.Tensor | None = torch.tensor(0.3),
        noise_parameter: torch.Tensor | None = torch.tensor(1.0),
        AC_MAX_VAL: int = -1,
        flag_additional_outputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Replace latents in CoolChicEncoder.
        self.size_per_latent = [lat.shape for lat in latents]
        # Something like self.latent_grids = nn.ParameterList(latents)
        # would break the computation graph. This doesn't. Following tips in:
        # https://github.com/qu-gg/torch-hypernetwork-tutorials?tab=readme-ov-file#tensorVSparameter
        for i in range(len(self.latent_grids)):
            del self.latent_grids[i].data
            self.latent_grids[i].data = latents[i]

        # This makes synthesis and arm happen with deltas added to the filters.
        if synth_delta is not None:
            self.synthesis.add_delta(
                synth_delta, add_to_weight=False, bias_only=self.only_delta_biases
            )
        if arm_delta is not None:
            self.arm.add_delta(
                arm_delta, add_to_weight=False, bias_only=self.only_delta_biases
            )

        # Forward pass (latents are in the class already).
        return super().forward(
            quantizer_noise_type=quantizer_noise_type,
            quantizer_type=quantizer_type,
            soft_round_temperature=soft_round_temperature,
            noise_parameter=noise_parameter,
            AC_MAX_VAL=AC_MAX_VAL,
            flag_additional_outputs=flag_additional_outputs,
        )

    def as_coolchic(
        self,
        latents: list[torch.Tensor],
        synth_delta: list[torch.Tensor] | None,
        arm_delta: list[torch.Tensor] | None,
        stop_grads: bool = False,
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

        encoder.synthesis.set_hypernet_weights(
            state_dict_to_param(self.synthesis.state_dict())
        )
        encoder.arm.set_hypernet_weights(state_dict_to_param(self.arm.state_dict()))
        encoder.upsampling.set_hypernet_weights(
            state_dict_to_param(self.upsampling.state_dict())
        )
        # Replace latents in CoolChicEncoder.
        encoder.size_per_latent = [lat.shape for lat in latents]

        # Only because stop grads is True.
        latents = [nn.Parameter(lat) for lat in latents]
        # Just checking.
        assert all(
            lat.is_leaf for lat in latents
        ), "Latents are not leaves. They still carry gradients back."
        synth_delta = (
            [nn.Parameter(delta) for delta in synth_delta]
            if synth_delta is not None
            else None
        )
        arm_delta = (
            [nn.Parameter(delta) for delta in arm_delta]
            if arm_delta is not None
            else None
        )

        # Something like self.latent_grids = nn.ParameterList(latents)
        # would break the computation graph. This doesn't. Following tips in:
        # https://github.com/qu-gg/torch-hypernetwork-tutorials?tab=readme-ov-file#tensorVSparameter
        for i in range(len(self.latent_grids)):
            del encoder.latent_grids[i].data
            encoder.latent_grids[i].data = latents[i]

        # This makes synthesis and arm happen with deltas added to the filters.
        if synth_delta is not None:
            encoder.synthesis.add_delta(
                synth_delta, add_to_weight=True, bias_only=self.only_delta_biases
            )
        if arm_delta is not None:
            encoder.arm.add_delta(
                arm_delta, add_to_weight=True, bias_only=self.only_delta_biases
            )

        return encoder.to(next(self.parameters()).device)

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
                None,  # synth_delta
                None,  # arm_delta
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


class NOWholeNet(WholeNet):
    def __init__(self, config: HyperNetConfig):
        super().__init__()
        self.config = config
        coolchic_encoder_parameter = CoolChicEncoderParameter(
            **get_coolchic_param_from_args(config.dec_cfg)
        )
        coolchic_encoder_parameter.set_image_size(config.patch_size)

        self.encoder = LatentHyperNet(
            n_latents=self.config.n_latents,
            n_hidden_channels=self.config.n_hidden_channels,
        )
        self.mean_decoder = LatentFreeCoolChicEncoder(param=coolchic_encoder_parameter)

    def forward(
        self,
        img: torch.Tensor,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "gaussian",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        softround_temperature: torch.Tensor = torch.tensor(0.3),
        noise_parameter: torch.Tensor = torch.tensor(0.25),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # input tensor is of the shape (batch_size, 3, H, W)
        latents = self.encoder.forward(img)
        # output is a list of tensors of the shape (batch_size, 1, H, W)
        assert latents[0].ndim == 4 and latents[0].shape[1] == 1

        return self.mean_decoder.forward(
            latents=latents,
            quantizer_noise_type=quantizer_noise_type,
            quantizer_type=quantizer_type,
            soft_round_temperature=softround_temperature,
            noise_parameter=noise_parameter,
            AC_MAX_VAL=-1,
            flag_additional_outputs=False,
        )

    def image_to_coolchic(
        self,
        img: torch.Tensor,
        stop_grads: bool = False,
    ) -> CoolChicEncoder:
        img = img.to(self.encoder.conv1ds[0].weight.device)
        if not stop_grads:
            raise NotImplementedError(
                "This method is only implemented for stop_grads=True."
            )

        self.eval()
        with torch.no_grad():
            latents = self.encoder.forward(img)
            cc_enc = self.mean_decoder.as_coolchic(
                latents=latents, stop_grads=stop_grads
            )
        self.train()
        return cc_enc

    def get_mlp_rate(self) -> float:
        # Get MLP rate.
        rate_mlp = 0.0
        rate_per_module = self.mean_decoder.get_network_rate()
        for _, module_rate in rate_per_module.items():  # pyright: ignore
            for _, param_rate in module_rate.items():  # weight, bias
                rate_mlp += param_rate
        return rate_mlp

    def freeze_resnet(self):
        """Not implemented."""
        pass

    def unfreeze_resnet(self):
        """Not implemented."""
        pass


class DeltaWholeNet(WholeNet):
    def __init__(self, config: HyperNetConfig):
        super().__init__()
        self.config = config
        coolchic_encoder_parameter = CoolChicEncoderParameter(
            **get_coolchic_param_from_args(config.dec_cfg)
        )
        coolchic_encoder_parameter.set_image_size(config.patch_size)

        self.hypernet = CoolchicHyperNet(config=config)
        self.mean_decoder = LatentFreeCoolChicEncoder(param=coolchic_encoder_parameter)

        self.use_delta = False

    def forward(
        self,
        img: torch.Tensor,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "gaussian",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        softround_temperature: torch.Tensor = torch.tensor(0.3),
        noise_parameter: torch.Tensor = torch.tensor(0.25),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.use_delta:
            latents, s_delta_dict, arm_delta_dict = self.hypernet.forward(img)
        else:
            latents = self.hypernet.latent_forward(img)
            s_delta_dict = {}
            arm_delta_dict = {}

        # Combine deltas with the parameters of the decoder.
        forward_params = self.add_deltas(
            s_delta_dict,
            arm_delta_dict,
            batch_size=latents[0].shape[0],
        )

        def get_forward_pass(lats: list[torch.Tensor], params: dict[str, torch.Tensor]):
            return functional_call(
                self.mean_decoder,
                params,
                (lats,),
                kwargs={
                    "quantizer_noise_type": quantizer_noise_type,
                    "quantizer_type": quantizer_type,
                    "soft_round_temperature": softround_temperature,
                    "noise_parameter": noise_parameter,
                    "AC_MAX_VAL": -1,
                    "flag_additional_outputs": False,
                },
            )

        # We need to add an additional batch dimension to latents so that after
        # vmap they still have a singleton batch dimension.
        latents = [lat.unsqueeze(1) for lat in latents]

        raw_out, rate, additional_data = torch.vmap(
            get_forward_pass, randomness="different"
        )(latents, forward_params)
        return raw_out.squeeze(1), rate.squeeze(1), additional_data

    def add_deltas(
        self,
        synth_delta_dict: dict[str, torch.Tensor],
        arm_delta_dict: dict[str, torch.Tensor],
        batch_size: int,
        remove_batch_dim: bool = False,
    ) -> dict[str, torch.Tensor]:
        if remove_batch_dim and batch_size != 1:
            raise ValueError(
                "Batch size should be 0 if we want to remove batch dimension."
            )

        # Adding deltas.
        forward_params: dict[str, torch.Tensor] = {}
        for k, v in self.mean_decoder.named_parameters():
            if (inner_key := k.removeprefix("synthesis.")) in synth_delta_dict:
                forward_params[k] = synth_delta_dict[inner_key] + v
            elif (inner_key := k.removeprefix("arm.")) in arm_delta_dict:
                forward_params[k] = arm_delta_dict[inner_key] + v
            else:
                forward_params[k] = v.unsqueeze(0).expand(batch_size, *v.shape)

            if remove_batch_dim:
                forward_params[k] = forward_params[k].squeeze(0)

        return forward_params

    def image_to_coolchic(
        self, img: torch.Tensor, stop_grads: bool = False
    ) -> CoolChicEncoder:
        img = img.to(next(self.parameters()).device)
        if self.use_delta:
            latents, s_delta_dict, arm_delta_dict = self.hypernet.forward(img)
        else:
            latents = self.hypernet.latent_forward(img)
            s_delta_dict = {}
            arm_delta_dict = {}

        # Combine deltas with the parameters of the decoder.
        forward_params = self.add_deltas(
            s_delta_dict,
            arm_delta_dict,
            batch_size=latents[0].shape[0],
            remove_batch_dim=True,
        )

        return self.mean_decoder.as_coolchic(
            latents=latents, stop_grads=stop_grads, new_parameters=forward_params
        )

    def get_mlp_rate(self) -> float:
        # Get MLP rate.
        rate_mlp = 0.0
        rate_per_module = self.mean_decoder.get_network_rate()
        for _, module_rate in rate_per_module.items():  # pyright: ignore
            for _, param_rate in module_rate.items():  # weight, bias
                rate_mlp += param_rate
        return rate_mlp

    def freeze_resnet(self):
        for param in self.hypernet.hn_backbone.parameters():
            param.requires_grad = False
        # for name, param in self.hypernet.latent_backbone.named_parameters():
        #     if "conv1" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    def unfreeze_resnet(self):
        for param in self.hypernet.hn_backbone.parameters():
            param.requires_grad = True
        # for param in self.hypernet.latent_backbone.parameters():
        #     param.requires_grad = True
        # Activate delta hypernet.
        self.use_delta = True

    def load_from_no_coolchic(self, no_coolchic: NOWholeNet) -> None:
        # Check that both models are on the same device.
        no_device = next(no_coolchic.parameters()).device
        self_device = next(self.parameters()).device
        if no_device != self_device:
            raise ValueError(
                f"Models are not on the same device. NOCoolChic: {no_device}, DeltaWholeNet: {self_device}"
            )

        # load state dict normally.
        self.mean_decoder.load_state_dict(no_coolchic.mean_decoder.state_dict())
        self.hypernet.latent_hn.load_state_dict(no_coolchic.encoder.state_dict())

        # we want deltas to be trainable.
        self.use_delta = True
        # and N-O coolchic weights shouldn't be trainable.
        for param in self.mean_decoder.parameters():
            param.requires_grad = False
        for param in self.hypernet.latent_hn.parameters():
            param.requires_grad = False

        # Initialize deltas so that their original output is
        # zero. (Output initially is the same as the NO CoolChic model
        # we initialized with).
        self.hypernet.init_deltas()

        # Check outputs are the same. Otherwise it means it wasn't loaded properly.
        img = torch.randn(1, 3, 256, 256)
        img = img.to(next(no_coolchic.parameters()).device)
        no_output, _, _ = no_coolchic.forward(
            img,
            quantizer_noise_type="none",
            quantizer_type="hardround",
        )
        img = img.to(next(self.parameters()).device)
        output, _, _ = self.forward(
            img,
            quantizer_noise_type="none",
            quantizer_type="hardround",
        )
        # Move to CPU to compare.
        no_output = no_output.cpu()
        output = output.cpu()
        if not torch.allclose(no_output, output, atol=1e-4):
            mse = torch.nn.functional.mse_loss(no_output, output).item()
            print(
                f"Outputs are not the same. MSE: {mse}. This means the model was not loaded properly."
            )
        else:
            print(
                "Outputs in the NO CoolChic checkpoints "
                "and the hypernet created from it match as expected."
            )


class SmallDeltaWholeNet(DeltaWholeNet):
    def __init__(self, config: HyperNetConfig):
        super().__init__(config)
        self.config = config
        coolchic_encoder_parameter = CoolChicEncoderParameter(
            **get_coolchic_param_from_args(config.dec_cfg)
        )
        coolchic_encoder_parameter.set_image_size(config.patch_size)

        self.hypernet = SmallCoolchicHyperNet(config=config)
        self.mean_decoder = LatentFreeCoolChicEncoder(param=coolchic_encoder_parameter)

        self.use_delta = False

    def freeze_resnet(self):
        """We don't want to freeze the backbone when using the small hypernet."""
        pass

    def unfreeze_resnet(self):
        """We don't want to unfreeze the backbone when using the small hypernet."""
        pass


class SmallAdditiveDeltaWholeNet(SmallDeltaWholeNet):
    def __init__(self, config: HyperNetConfig):
        super().__init__(config)

        # Modify the prediction heads so that they only output biases.
        self.hypernet = SmallAdditiveHyperNet(config=config)
        coolchic_encoder_parameter = CoolChicEncoderParameter(
            **get_coolchic_param_from_args(config.dec_cfg)
        )
        coolchic_encoder_parameter.set_image_size(config.patch_size)
        self.mean_decoder = LatentFreeCoolChicEncoder(param=coolchic_encoder_parameter)
        self.use_delta = False

import torch
from pydantic import BaseModel
from torch import nn
from torchvision.models import resnet50

from coolchic.hypernet.common import ResidualBlockDown, build_mlp
from coolchic.utils.types import DecoderConfig


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


ConvBackbone = resnet50

BACKBONE_OUTPUT_FEATURES = 2048


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

    def n_params_synthesis(self) -> int:
        """Calculates the number of parameters needed for the synthesis network."""
        n_params = 0
        input_ft = self.n_latents  # The input to synthesis are the upsampled latents.
        # There is one layer per line in layers_dim. We need to decode it to get the number of parameters.
        for layers in self.layers_dim:
            out_ft, k_size, _, _ = layers.split("-")
            out_ft = int(out_ft)
            k_size = int(k_size)

            n_params += self.n_params_synthesis_layer(input_ft, out_ft, k_size)
            input_ft = out_ft
        return n_params

    @staticmethod
    def n_params_synthesis_layer(in_channels: int, out_channels: int, k: int) -> int:
        """Every synthesis layer has out_channels conv kernels of size in_channels x k x k."""
        return in_channels * out_channels * k * k


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
        self.n_output_features = self.n_params_synthesis()

        # The layers we need: an MLP with hypernet_n_layers hidden layers.
        self.hidden_size = hypernet_hidden_dim

        self.mlp = build_mlp(
            input_size=self.n_input_features,
            output_size=self.n_output_features,
            n_hidden_layers=hypernet_n_layers,
            hidden_size=self.hidden_size,
        )

    def n_params_synthesis(self) -> int:
        """Calculates the number of parameters needed for the arm network.
        An arm network is an MLP with n_hidden_layers of size dim_arm->dim_arm.
        The output layer outputs 2 values (mu and sigma)."""
        return self.dim_arm * self.dim_arm * self.n_hidden_layers + self.dim_arm * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


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


class HyperNetParams(BaseModel):
    hidden_dim: int
    n_layers: int


class HyperNetConfig(BaseModel):
    dec_cfg: DecoderConfig

    synthesis: HyperNetParams = HyperNetParams(hidden_dim=1024, n_layers=3)
    arm: HyperNetParams = HyperNetParams(hidden_dim=1024, n_layers=3)
    upsampling: HyperNetParams = HyperNetParams(hidden_dim=256, n_layers=1)

    def model_post_init(self, _) -> None:
        self.n_latents = len(self.dec_cfg.parsed_n_ft_per_res)


class CoolchicHyperNet(nn.Module):
    def __init__(self, config: HyperNetConfig) -> None:
        super().__init__()
        self.config = config

        # Instantiate all the hypernetworks.
        self.latent_hn = LatentHyperNet(n_latents=self.config.n_latents)
        self.hn_backbone = ConvBackbone()
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

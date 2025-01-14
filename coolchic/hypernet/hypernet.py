import torch
from pydantic import BaseModel
from torch import nn

from coolchic.hypernet.arm import ArmHyperNet
from coolchic.hypernet.backbone import ConvBackbone
from coolchic.hypernet.latent import LatentHyperNet
from coolchic.hypernet.synthesis import SynthesisHyperNet
from coolchic.hypernet.upsampling import UpsamplingHyperNet
from coolchic.utils.types import DecoderConfig


class HyperNetParams(BaseModel):
    hidden_dim: int
    n_layers: int


class HyperNetConfig(BaseModel):
    dec_cfg: DecoderConfig

    synthesis: HyperNetParams = HyperNetParams(hidden_dim=1024, n_layers=3)
    arm: HyperNetParams = HyperNetParams(hidden_dim=1024, n_layers=3)
    upsampling: HyperNetParams = HyperNetParams(hidden_dim=256, n_layers=1)

    def model_post_init(self, __context) -> None:
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

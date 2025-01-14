import torch
from torch import nn

from coolchic.hypernet.backbone import BACKBONE_OUTPUT_FEATURES
from coolchic.hypernet.common import build_mlp


def symmetric_filter_n_params_from_target(target_size: int) -> int:
    """Calculates the number of parameters needed for a symmetric filter
    of size target_size.

    For a kernel of size target_k_size = 2N, we need N values
    e.g. 3 params a b c to parameterize a b c c b a.
    For a kernel of size target_k_size = 2N + 1, we need N + 1 values
    e.g. 4 params a b c d to parameterize a b c d c b a.
    """
    return (target_size + 1) // 2


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
        self.ups_n_params = symmetric_filter_n_params_from_target(ups_k_size)
        self.ups_preconcat_k_size = ups_preconcat_k_size
        self.ups_preconcat_n_params = symmetric_filter_n_params_from_target(
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

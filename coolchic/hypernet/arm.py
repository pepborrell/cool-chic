import torch
from torch import nn

from coolchic.hypernet.backbone import BACKBONE_OUTPUT_FEATURES
from coolchic.hypernet.common import build_mlp


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

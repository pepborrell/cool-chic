import torch
from torch import nn


def build_mlp(
    input_size: int,
    output_size: int,
    n_hidden_layers: int,
    hidden_size: int,
    activation: nn.Module = nn.ReLU(),
) -> nn.Module:
    """Builds an MLP with n_hidden_layers hidden layers."""
    layers_list = nn.ModuleList()
    # Start with the input layer.
    layers_list.append(nn.Linear(input_size, hidden_size))
    layers_list.append(activation)

    # Then the hidden layers.
    for _ in range(n_hidden_layers):
        layers_list.append(nn.Linear(hidden_size, hidden_size))
        layers_list.append(activation)

    # Add output layer.
    layers_list.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers_list)


class ConvNextBlock(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.dw_conv = nn.Conv2d(
            self.n_channels, self.n_channels, kernel_size=7, groups=self.n_channels
        )
        self.conv1 = nn.Conv2d(
            self.n_channels, self.n_channels, kernel_size=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            self.n_channels, self.n_channels, kernel_size=1, padding=0
        )
        self.layer_norm = nn.LayerNorm(self.n_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dw_conv(x)
        z = self.layer_norm(z)
        z = self.conv1(z)
        z = self.gelu(z)
        z = self.conv2(z)
        return z + x  # Residual connection


class ResidualBlockDown(nn.Module):
    """A residual block based on ConvNext with optional downsampling.
    As described in Overfitted image coding at reduced complexity, Blard et al.
    """

    def __init__(
        self, in_channels: int, out_channels: int | None = None, downsample_n: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = (
            out_channels if out_channels is not None else self.in_channels
        )
        self.downsample_n = downsample_n

        self.strided_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.downsample_n,
            padding=1,
        )
        self.conv1d = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=1, padding=0
        )
        self.convnext_pre = ConvNextBlock(self.out_channels)
        self.convnexts_post = nn.Sequential(
            ConvNextBlock(self.out_channels), ConvNextBlock(self.out_channels)
        )

        self.layer_norm = nn.LayerNorm(self.out_channels)
        self.gelu = nn.GELU()
        self.avg_pool = nn.AvgPool2d(2, stride=self.downsample_n, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1
        z = self.strided_conv(x)
        z = self.layer_norm(z)
        z = self.gelu(z)
        z = self.convnext_pre(z)
        # Branch 2
        y = self.avg_pool(x)
        y = self.conv1d(y)
        # Merge branches
        z = z + y
        z = self.convnexts_post(z)
        return z

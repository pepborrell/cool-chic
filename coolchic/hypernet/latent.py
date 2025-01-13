import torch
from torch import nn


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

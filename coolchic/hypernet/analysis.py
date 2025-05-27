import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch import Tensor


class LayerNorm2d(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        input = input.permute(0, 2, 3, 1)
        input = F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )
        input = input.permute(0, 3, 1, 2)
        return input


class Block(nn.Module):
    """ConvNeXt block"""

    def __init__(self, nf, ks=7, layer_scale=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(nf, nf, kernel_size=ks, padding=ks // 2, groups=nf)
        self.norm = LayerNorm2d(nf, eps=1e-6)
        self.pwconv1 = nn.Conv2d(nf, nf * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(nf * 4, nf, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.ones(nf, 1, 1) * layer_scale)

    def forward(self, x: Tensor):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale * x + shortcut
        return x


class ResidualBlock(nn.Module):
    def __init__(self, ni: int, nf: int, n_blocks=2, stride=1):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(ni, nf, kernel_size=3, padding=1, stride=stride),
            LayerNorm2d(nf, eps=1e-6),
            nn.GELU(),
            Block(nf),
        )
        self.identity = nn.Sequential(
            (
                nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True)
                if stride > 1
                else nn.Identity()
            ),
            nn.Conv2d(ni, nf, kernel_size=1),
        )
        # self.identity = nn.Conv2d(ni, nf, kernel_size=1, stride=stride)
        self.residual = nn.Sequential(
            *[Block(nf) for _ in range(n_blocks)],
        )

    def forward(self, x: Tensor):
        x = self.downsample(x) + self.identity(x)
        x = self.residual(x)
        return x


class Analysis(nn.Module):
    def __init__(self, in_channels=3, n_grids=7, n_feat=64, n_blocks=2):
        super().__init__()
        self.n_feat = n_feat

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(in_channels, n_feat, n_blocks),
                *[
                    ResidualBlock(n_feat, n_feat, n_blocks, stride=2)
                    for _ in range(n_grids - 1)
                ],
            ]
        )

        self.fuses = nn.ModuleList(
            [nn.Conv2d(n_feat, 1, kernel_size=1) for _ in range(n_grids)]
        )

        self.reinitialize_parameters()

        # For flop analysis.
        self.flops_per_module = {k: 0 for k in ["blocks", "fuses"]}

    def reinitialize_parameters(self) -> None:
        for m in self.children():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> list[Tensor]:
        grids = []
        for block, fuse in zip(self.blocks, self.fuses):
            x = block(x)
            grids.append(fuse(x))
        return grids

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

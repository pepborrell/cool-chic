from typing import Literal, OrderedDict

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch import nn


def build_mlp(
    input_size: int,
    output_size: int,
    n_hidden_layers: int,
    hidden_size: int,
    activation: nn.Module = nn.ReLU(),
    output_activation: str | None = None,
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
    if output_activation is not None:
        OUTPUT_ACTIVATION_DICT: dict[str, nn.Module] = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.2),
        }
        try:
            out_act_module = OUTPUT_ACTIVATION_DICT[output_activation]
        except KeyError:
            raise ValueError(
                f"Output activation {output_activation} not supported. "
                f"Supported activations are: {list(OUTPUT_ACTIVATION_DICT.keys())}"
            )
        layers_list.append(out_act_module)
    return nn.Sequential(*layers_list)


class ConvNextBlock(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        ### ALL LAYERS ###
        self.dw_conv = nn.Conv2d(
            self.n_channels,
            self.n_channels,
            kernel_size=7,
            groups=self.n_channels,
            padding="same",
            bias=True,
        )
        self.conv1 = nn.Conv2d(
            self.n_channels, self.n_channels, kernel_size=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            self.n_channels, self.n_channels, kernel_size=1, padding=0, bias=False
        )
        self.layer_norm = nn.GroupNorm(num_groups=1, num_channels=self.n_channels)
        self.gelu = nn.GELU()

        ### INITIALIZATION ###
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.dw_conv.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in")
        assert self.dw_conv.bias is not None, "dw_conv bias is None."
        nn.init.zeros_(self.dw_conv.bias)
        # conv1 and conv2 have no bias.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input has size (B, C, H, W)
        z = self.dw_conv(x)
        z = self.layer_norm(z)
        z = self.conv1(z)
        z = self.gelu(z)
        z = self.conv2(z)  # (B, C, H, W)
        return z + x  # Residual connection


CONV_NEXT_BLOCK_SCALE_INIT = 1e-6


class ConvNextBlockGamma(ConvNextBlock):
    def __init__(self, n_channels: int) -> None:
        super().__init__(n_channels)
        self.gamma = nn.Parameter(
            torch.full((self.n_channels,), CONV_NEXT_BLOCK_SCALE_INIT)
        )  # Scale parameter for the residual connection

        ### INITIALIZATION ###
        self._init_weights()

    def _init_weights(self):
        super()()._init_weights()
        # Initialize the scale parameter
        nn.init.constant_(self.gamma, CONV_NEXT_BLOCK_SCALE_INIT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input has size (B, C, H, W)
        z = self.dw_conv(x)
        z = self.layer_norm(z)
        z = self.conv1(z)
        z = self.gelu(z)
        z = self.conv2(z)  # (B, C, H, W)
        # Scale the output.
        z = (
            self.gamma.view(1, self.n_channels, 1, 1) * z
        )  # Broadcasting to make multiplication work.
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
            self.in_channels, self.out_channels, kernel_size=1, padding=0
        )
        self.convnext_pre = ConvNextBlock(self.out_channels)
        self.convnexts_post = nn.Sequential(
            ConvNextBlock(self.out_channels), ConvNextBlock(self.out_channels)
        )

        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
        self.gelu = nn.GELU()
        # In the non-downsampling case, padding is applied manually and only on the left.
        self.pre_pool_padding = (
            lambda x: nn.functional.pad(x, (1, 0, 1, 0))
            if self.downsample_n == 1
            else x
        )
        self.avg_pool = nn.AvgPool2d(
            2, stride=self.downsample_n, padding=0, ceil_mode=True
        )

        self.reset_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1
        z = self.strided_conv(x)
        z = self.group_norm(z)
        z = self.gelu(z)
        z = self.convnext_pre(z)
        # Branch 2
        y = self.pre_pool_padding(x)
        y = self.avg_pool(y)
        y = self.conv1d(y)
        # Merge branches
        z = z + y
        z = self.convnexts_post(z)
        return z

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize the weights as done in the ConvNeXt paper.
        https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L103
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, ConvNextBlock):
            m._init_weights()

    def reset_weights(self):
        """Reset the weights of the model."""
        self.apply(self._init_weights)


def select_param_from_name(obj: nn.Module, name: str) -> tuple[torch.Tensor, nn.Module]:
    """Select a parameter from a module by name, where the name
    follows how parametes are usually named inside of nn modules.
    """
    split_name = name.split(".")
    obj_parent = None
    for sub_name in split_name:
        if sub_name.isdigit():
            obj_parent = obj
            assert hasattr(obj, "__getitem__"), "Selected object is not a container."
            obj = obj[int(sub_name)]  # pyright: ignore
        else:
            obj_parent = obj
            obj = getattr(obj, sub_name)
    assert isinstance(obj, torch.Tensor), (
        "Selected object is not a tensor. " f"Got {type(obj)} instead."
    )
    assert isinstance(obj_parent, nn.Module), "Selected object has no parent."
    return obj, obj_parent


def set_hypernet_weights(
    obj,
    all_weights: OrderedDict[str, torch.Tensor] | OrderedDict[str, torch.nn.Parameter],
):
    """Set the weights coming from the hypernetwork.
    The weights are copied so that all gradients can flow through them.
    """
    for name, weight in all_weights.items():
        _, weight_parent = select_param_from_name(obj, name)
        if name.endswith("weight"):
            del weight_parent.weight
            weight_parent.weight = weight
        elif name.endswith("bias"):
            del weight_parent.bias
            weight_parent.bias = weight
        elif name.endswith(
            "original"
        ):  # transpose convolutions are using `parametrize`.
            del weight_parent.original
            weight_parent.original = weight
        else:
            raise ValueError(f"Unknown parameter name {name}")


def upsample_latents(
    latents: list[torch.Tensor],
    mode: Literal["nearest", "linear", "bilinear", "bicubic"],
    img_size: tuple[int, int],
):
    assert all(lat.ndim == 4 for lat in latents), "All latents must be 4D tensors."
    return torch.cat(
        [
            torch.nn.functional.interpolate(lat, size=img_size, mode=mode)
            for lat in latents
        ],
        dim=1,
    )


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

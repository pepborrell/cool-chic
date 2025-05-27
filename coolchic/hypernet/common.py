from typing import Iterator, Literal, OrderedDict

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


class Block(nn.Module):
    def __init__(self, n_channels: int, layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        self.n_channels = n_channels
        ### ALL LAYERS ###
        self.dwconv = nn.Conv2d(
            self.n_channels,
            self.n_channels,
            kernel_size=7,
            groups=self.n_channels,
            padding="same",
            bias=True,
        )
        self.pwconv1 = nn.Conv2d(
            self.n_channels, self.n_channels * 4, kernel_size=1, padding=0
        )
        self.pwconv2 = nn.Conv2d(
            self.n_channels * 4, self.n_channels, kernel_size=1, padding=0
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.n_channels, eps=1e-6)
        self.gelu = nn.GELU()
        self.layer_scale = nn.Parameter(
            torch.ones(self.n_channels, 1, 1) * layer_scale_init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input has size (B, C, H, W)
        z = self.dwconv(x)
        z = self.norm(z)
        z = self.pwconv1(z)
        z = self.gelu(z)
        z = self.pwconv2(z)  # (B, C, H, W)
        return self.layer_scale * z + x  # Residual connection


class ResidualBlock(nn.Module):
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

        # Declare branch 1's components.
        self.downsample = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                stride=self.downsample_n,
                padding=1,
            ),
            nn.GroupNorm(num_groups=1, num_channels=self.out_channels),
            nn.GELU(),
            Block(self.out_channels),
        )

        # Branch 2.
        self.identity = nn.Sequential(
            nn.AvgPool2d(2, stride=self.downsample_n, ceil_mode=True)
            if self.downsample_n > 1
            # If no downsampling is needed, no need to pool.
            else nn.Identity(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0),
        )

        # Elements used after the merge.
        self.residual = nn.Sequential(
            Block(self.out_channels), Block(self.out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.downsample(x)
        y = self.identity(x)
        return self.residual(z + y)


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


def add_deltas(
    cc_params: Iterator[tuple[str, nn.Parameter]],
    synth_delta_dict: dict[str, torch.Tensor],
    arm_delta_dict: dict[str, torch.Tensor],
    batch_size: int,
    remove_batch_dim: bool = False,
) -> dict[str, torch.Tensor]:
    if remove_batch_dim and batch_size != 1:
        raise ValueError("Batch size should be 0 if we want to remove batch dimension.")

    # Adding deltas.
    forward_params: dict[str, torch.Tensor] = {}
    for k, v in cc_params:
        if (inner_key := k.removeprefix("synthesis.")) in synth_delta_dict:
            forward_params[k] = synth_delta_dict[inner_key] + v
        elif (inner_key := k.removeprefix("arm.")) in arm_delta_dict:
            forward_params[k] = arm_delta_dict[inner_key] + v
        else:
            forward_params[k] = v.unsqueeze(0).expand(batch_size, *v.shape)

        if remove_batch_dim:
            forward_params[k] = forward_params[k].squeeze(0)

    return forward_params

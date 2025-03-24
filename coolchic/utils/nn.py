from typing import cast

import torch
from torch import nn

from coolchic.enc.component.coolchic import CoolChicEncoder


def get_num_of_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Custom scheduling function for the soft rounding temperature and the noise parameter
# Taken from coolchic/enc/training/train.py
def _linear_schedule(
    initial_value: float, final_value: float, cur_itr: float, max_itr: float
) -> float:
    """Linearly schedule a function to go from initial_value at cur_itr = 0 to
    final_value when cur_itr = max_itr.

    Args:
        initial_value (float): Initial value for the scheduling
        final_value (float): Final value for the scheduling
        cur_itr (float): Current iteration index
        max_itr (float): Total number of iterations

    Returns:
        float: The linearly scheduled value @ iteration number cur_itr
    """
    assert cur_itr >= 0 and cur_itr <= max_itr, (
        f"Linear scheduling from 0 to {max_itr} iterations"
        " except to have a current iterations between those two values."
        f" Found cur_itr = {cur_itr}."
    )

    return cur_itr * (final_value - initial_value) / max_itr + initial_value


def get_mlp_rate(net: CoolChicEncoder) -> float:
    rate_mlp = 0.0
    rate_per_module = net.get_network_rate()
    rate_per_module = cast(dict[str, dict[str, float]], rate_per_module)  # for pyright

    for _, module_rate in rate_per_module.items():
        for _, param_rate in module_rate.items():  # weight, bias
            rate_mlp += param_rate

    if rate_mlp == 0.0:
        raise ValueError("Model has no parameters quantized parameters.")
    rate_mlp = rate_mlp.item() if isinstance(rate_mlp, torch.Tensor) else rate_mlp
    return rate_mlp

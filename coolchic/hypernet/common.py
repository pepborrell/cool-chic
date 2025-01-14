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

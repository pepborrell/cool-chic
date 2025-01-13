from torch import nn

from coolchic.hypernet.backbone import BACKBONE_OUTPUT_FEATURES


class SynthesisHyperNet(nn.Module):
    """Takes a latent tensor and outputs the filters of the synthesis network."""

    def __init__(self, n_latents: int, layers_dim: list[str]) -> None:
        super().__init__()
        self.n_input_features = BACKBONE_OUTPUT_FEATURES

        self.n_latents = n_latents
        self.layers_dim = layers_dim

        # For hop config, this will be 642 parameters.
        self.n_output_features = self.n_params_synthesis()

        # The layers we need: an MLP with 3 hidden layers.
        self.hidden_size = 1024
        self.mlp = nn.Sequential(
            nn.Linear(self.n_input_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.n_output_features),
        )

    def n_params_synthesis(self) -> int:
        """Calculates the number of parameters needed for the synthesis network."""
        n_params = 0
        input_ft = self.n_latents  # The input to synthesis are the upsampled latents.
        # There is one layer per line in layers_dim. We need to decode it to get the number of parameters.
        for layers in self.layers_dim:
            out_ft, k_size, _, _ = layers.split("-")
            out_ft = int(out_ft)
            k_size = int(k_size)

            n_params += self.n_params_synthesis_layer(input_ft, out_ft, k_size)
            input_ft = out_ft
        return n_params

    @staticmethod
    def n_params_synthesis_layer(in_channels: int, out_channels: int, k: int) -> int:
        """Every synthesis layer has out_channels conv kernels of size in_channels x k x k."""
        return in_channels * out_channels * k * k
